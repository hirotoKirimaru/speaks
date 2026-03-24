"""speaks - ローカル音声文字起こし + 議事録要約ツール"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer
from faster_whisper import WhisperModel

app = typer.Typer(help="音声ファイルを文字起こし + 議事録要約")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
WHISPER_MODEL = "large-v3"

SUMMARY_PROMPT = """以下は会議や雑談の文字起こしです。これを議事録形式にまとめてください。

## 出力フォーマット（必ずこの形式で）

# 議事録

## トピック
1. （話題を箇条書き）

## 決定事項
- （決まったこと。なければ「特になし」）

## TODO
- [ ] （やるべきこと。なければ「特になし」）

## 議論メモ
- （話の流れや重要な発言を要約）

---

## 文字起こし原文:

{transcript}
"""


def _resolve_hf_token() -> str:
    """HFトークンを自動解決: 環境変数 → ~/.cache/huggingface/token"""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""


def _diarize(audio_path: str, hf_token: str) -> list[tuple[float, float, str]]:
    """pyannote で話者分離"""
    print("[1/3] 話者分離中...", file=sys.stderr)
    start = time.time()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    result = pipeline(audio_path)

    turns = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))

    elapsed = time.time() - start
    speakers = sorted(set(t[2] for t in turns))
    print(f"    完了 ({elapsed:.1f}秒, 話者数: {len(speakers)})", file=sys.stderr)
    return turns


def _find_speaker(timestamp: float, turns: list[tuple[float, float, str]]) -> str:
    """タイムスタンプに対応する話者を返す"""
    for start, end, speaker in turns:
        if start <= timestamp <= end:
            return speaker
    min_dist = float("inf")
    closest = "不明"
    for start, end, speaker in turns:
        dist = min(abs(timestamp - start), abs(timestamp - end))
        if dist < min_dist:
            min_dist = dist
            closest = speaker
    return closest


def _transcribe(
    audio_path: str,
    model_size: str,
    speaker_turns: list[tuple[float, float, str]] | None = None,
) -> str:
    """faster-whisper で音声を文字起こし"""
    step = "[2/3]" if speaker_turns is not None else "[1/2]"
    print(f"{step} 文字起こし中... (model: {model_size})", file=sys.stderr)
    start = time.time()

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(audio_path, language="ja", beam_size=5)
    segments = list(segments)

    elapsed = time.time() - start
    print(
        f"    完了 ({elapsed:.1f}秒, 言語: {info.language}, 確率: {info.language_probability:.0%})",
        file=sys.stderr,
    )

    if speaker_turns is None:
        lines = []
        for seg in segments:
            lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        return "\n".join(lines)

    lines = []
    for seg in segments:
        seg_mid = (seg.start + seg.end) / 2
        speaker = _find_speaker(seg_mid, speaker_turns)
        lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {speaker}: {seg.text}")
    return "\n".join(lines)


def _summarize(transcript: str, model: str, step: str = "[3/3]") -> str:
    """Ollama で議事録風に要約"""
    print(f"{step} 要約中... (model: {model})", file=sys.stderr)
    start = time.time()

    prompt = SUMMARY_PROMPT.format(transcript=transcript)

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300.0,
    )
    response.raise_for_status()
    result = response.json()["response"]

    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)
    return result


@app.command()
def run(
    audio: Annotated[Path, typer.Argument(help="入力WAVファイルのパス")],
    whisper_model: Annotated[str, typer.Option(help="Whisperモデル")] = WHISPER_MODEL,
    ollama_model: Annotated[str, typer.Option(help="Ollamaモデル")] = OLLAMA_MODEL,
    transcript_only: Annotated[bool, typer.Option("--transcript-only", help="文字起こしのみ")] = False,
    no_diarize: Annotated[bool, typer.Option("--no-diarize", help="話者分離を無効化")] = False,
    hf_token: Annotated[Optional[str], typer.Option(help="HuggingFaceトークン")] = None,
    output_dir: Annotated[Path, typer.Option(help="出力ディレクトリ")] = Path("output"),
):
    """音声ファイルを文字起こしして議事録を生成する。"""
    if not audio.exists():
        typer.echo(f"エラー: ファイルが見つかりません: {audio}", err=True)
        raise typer.Exit(1)

    token = hf_token or _resolve_hf_token()
    if not no_diarize and not token:
        typer.echo(
            "エラー: 話者分離には HuggingFace トークンが必要です。\n"
            "  huggingface-cli login でログインするか、\n"
            "  --hf-token <token> または環境変数 HF_TOKEN を設定してください。\n"
            "  話者分離なしで実行するには --no-diarize を指定してください。",
            err=True,
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    speaker_turns = None
    if not no_diarize:
        speaker_turns = _diarize(str(audio), token)

    transcript = _transcribe(str(audio), whisper_model, speaker_turns)

    transcript_path = output_dir / f"{prefix}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    typer.echo(f"  → {transcript_path}", err=True)

    if not transcript_only:
        step = "[3/3]" if not no_diarize else "[2/2]"
        summary = _summarize(transcript, ollama_model, step)
        minutes_path = output_dir / f"{prefix}_minutes.md"
        minutes_path.write_text(summary, encoding="utf-8")
        typer.echo(f"  → {minutes_path}", err=True)


if __name__ == "__main__":
    app()
