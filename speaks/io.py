"""共通定数とパスユーティリティ。"""

from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
WHISPER_MODEL = "large-v3"
TARGET_SR = 16000


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    """出力ファイル名の規約を 1 か所にまとめる。"""
    return {
        "transcript": output_dir / f"{prefix}_transcript.txt",
        "minutes": output_dir / f"{prefix}_minutes.md",
    }
