"""speaks - ローカル音声文字起こし + 議事録要約ツール"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from faster_whisper import WhisperModel


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
WHISPER_MODEL = "large-v3"

def _resolve_hf_token() -> str:
    """HFトークンを自動解決: 環境変数 → ~/.cache/huggingface/token"""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""

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


def diarize(audio_path: str, hf_token: str) -> list[tuple[float, float, str]]:
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


def transcribe(
    audio_path: str,
    model_size: str = WHISPER_MODEL,
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

    # 話者分離結果とマージ
    lines = []
    for seg in segments:
        seg_mid = (seg.start + seg.end) / 2
        speaker = _find_speaker(seg_mid, speaker_turns)
        lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {speaker}: {seg.text}")
    return "\n".join(lines)


def _find_speaker(
    timestamp: float, turns: list[tuple[float, float, str]]
) -> str:
    """タイムスタンプに対応する話者を返す"""
    for start, end, speaker in turns:
        if start <= timestamp <= end:
            return speaker
    # 最も近い区間の話者を返す
    min_dist = float("inf")
    closest = "不明"
    for start, end, speaker in turns:
        dist = min(abs(timestamp - start), abs(timestamp - end))
        if dist < min_dist:
            min_dist = dist
            closest = speaker
    return closest


def summarize(transcript: str, model: str = OLLAMA_MODEL) -> str:
    """Ollama で議事録風に要約"""
    print("[3/3] 要約中..." if True else "[2/2] 要約中...", file=sys.stderr)
    print(f"  (model: {model})", file=sys.stderr)
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


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルを文字起こし + 議事録要約"
    )
    parser.add_argument("audio", help="入力WAVファイルのパス")
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisperモデル (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help=f"Ollamaモデル (default: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="文字起こしのみ（要約なし）",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="話者分離を無効化",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace トークン (省略時は ~/.cache/huggingface/token or HF_TOKEN)",
    )
    parser.add_argument(
        "--output-dir", default="output", help="出力ディレクトリ (default: output)"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"エラー: ファイルが見つかりません: {audio_path}", file=sys.stderr)
        sys.exit(1)

    hf_token = args.hf_token or _resolve_hf_token()
    if not args.no_diarize and not hf_token:
        print(
            "エラー: 話者分離には HuggingFace トークンが必要です。\n"
            "  huggingface-cli login でログインするか、\n"
            "  --hf-token <token> または環境変数 HF_TOKEN を設定してください。\n"
            "  話者分離なしで実行するには --no-diarize を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # 話者分離
    speaker_turns = None
    if not args.no_diarize:
        speaker_turns = diarize(str(audio_path), hf_token)

    # 文字起こし
    transcript = transcribe(str(audio_path), args.whisper_model, speaker_turns)

    transcript_path = output_dir / f"{prefix}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"  → {transcript_path}", file=sys.stderr)

    # 要約
    if not args.transcript_only:
        summary = summarize(transcript, args.ollama_model)
        minutes_path = output_dir / f"{prefix}_minutes.md"
        minutes_path.write_text(summary, encoding="utf-8")
        print(f"  → {minutes_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
