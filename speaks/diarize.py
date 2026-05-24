"""話者分離 (pyannote 3.1)。HF トークン解決もここに集約する。"""

import os
import sys
import time
from pathlib import Path


SpeakerTurns = list[tuple[float, float, str]]


def resolve_hf_token() -> str:
    """HFトークンを自動解決: 環境変数 → ~/.cache/huggingface/token"""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""


def diarize(audio_path: str, hf_token: str) -> SpeakerTurns:
    """pyannote で話者分離"""
    print("[1/3] 話者分離中...", file=sys.stderr)
    start = time.time()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    result = pipeline(audio_path)

    turns: SpeakerTurns = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))

    elapsed = time.time() - start
    speakers = sorted(set(t[2] for t in turns))
    print(f"    完了 ({elapsed:.1f}秒, 話者数: {len(speakers)})", file=sys.stderr)
    return turns


def find_speaker(timestamp: float, turns: SpeakerTurns) -> str:
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
