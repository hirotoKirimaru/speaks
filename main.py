"""speaks - ローカル音声文字起こし + 議事録要約ツール"""

import argparse
import sys
import time
from pathlib import Path

import httpx
from faster_whisper import WhisperModel


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


def transcribe(audio_path: str, model_size: str = WHISPER_MODEL) -> str:
    """faster-whisper で音声を文字起こし"""
    print(f"[1/2] 文字起こし中... (model: {model_size})", file=sys.stderr)
    start = time.time()

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(audio_path, language="ja", beam_size=5)

    lines = []
    for segment in segments:
        timestamp = f"[{segment.start:.1f}s - {segment.end:.1f}s]"
        lines.append(f"{timestamp} {segment.text}")

    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒, 言語: {info.language}, 確率: {info.language_probability:.0%})", file=sys.stderr)
    return "\n".join(lines)


def summarize(transcript: str, model: str = OLLAMA_MODEL) -> str:
    """Ollama で議事録風に要約"""
    print(f"[2/2] 要約中... (model: {model})", file=sys.stderr)
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
    parser = argparse.ArgumentParser(description="音声ファイルを文字起こし + 議事録要約")
    parser.add_argument("audio", help="入力WAVファイルのパス")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"Whisperモデル (default: {WHISPER_MODEL})")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help=f"Ollamaモデル (default: {OLLAMA_MODEL})")
    parser.add_argument("--transcript-only", action="store_true", help="文字起こしのみ（要約なし）")
    parser.add_argument("--output", "-o", help="出力ファイルパス（省略時は標準出力）")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"エラー: ファイルが見つかりません: {audio_path}", file=sys.stderr)
        sys.exit(1)

    transcript = transcribe(str(audio_path), args.whisper_model)

    if args.transcript_only:
        output = transcript
    else:
        summary = summarize(transcript, args.ollama_model)
        output = f"{summary}\n\n---\n\n## 文字起こし全文\n\n{transcript}"

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"出力: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
