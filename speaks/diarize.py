"""話者分離: pyannote / Sortformer から選択可能なバックエンド。

論文の知見:
- pyannote 3.1: Powerset cross entropy (Plaquet & Bredin 2023) を採用。
- Sortformer (NVIDIA, arXiv:2507.18446): pyannote 比で DER 約半減という報告。
  本実装ではオフライン用途のラッパに留め、NeMo は optional extra。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Protocol

SpeakerTurns = list[tuple[float, float, str]]


class Diarizer(Protocol):
    def __call__(self, audio_path: str) -> SpeakerTurns: ...


def resolve_hf_token() -> str:
    """HFトークンを自動解決: 環境変数 → ~/.cache/huggingface/token"""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""


class PyannoteDiarizer:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token

    def __call__(self, audio_path: str) -> SpeakerTurns:
        print("[diarize] pyannote 3.1 で話者分離中...", file=sys.stderr)
        start = time.time()

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=self.hf_token
        )
        result = pipeline(audio_path)

        turns: SpeakerTurns = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        elapsed = time.time() - start
        speakers = sorted(set(t[2] for t in turns))
        print(
            f"    完了 ({elapsed:.1f}秒, 話者数: {len(speakers)})",
            file=sys.stderr,
        )
        return turns


class SortformerDiarizer:
    """NVIDIA Sortformer (offline) ラッパ。NeMo が必要なため optional。"""

    def __init__(self, model_id: str = "nvidia/diar_sortformer_4spk-v1"):
        self.model_id = model_id

    def __call__(self, audio_path: str) -> SpeakerTurns:
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise RuntimeError(
                "Sortformer を使うには NeMo が必要です。\n"
                "  uv pip install 'speaks[sortformer]'\n"
                "または `pip install nemo_toolkit[asr]` でインストールしてください。"
            ) from e

        print(f"[diarize] Sortformer ({self.model_id}) で話者分離中...", file=sys.stderr)
        start = time.time()

        model = SortformerEncLabelModel.from_pretrained(self.model_id)
        model.eval()
        # NeMo の Sortformer は predict() で speaker turn list を返す API を持つ。
        # 出力フォーマットは (start, end, speaker_label) のリストに揃える。
        result = model.diarize(audio=audio_path)
        turns: SpeakerTurns = []
        for entry in result[0] if result else []:
            # NeMo の出力は "start end speaker_label" 形式の文字列、または
            # tuple/list のどちらかで来ることがある。ここで両対応する。
            if isinstance(entry, str):
                parts = entry.split()
                if len(parts) >= 3:
                    turns.append((float(parts[0]), float(parts[1]), parts[2]))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                turns.append((float(entry[0]), float(entry[1]), str(entry[2])))

        elapsed = time.time() - start
        speakers = sorted(set(t[2] for t in turns))
        print(
            f"    完了 ({elapsed:.1f}秒, 話者数: {len(speakers)})",
            file=sys.stderr,
        )
        return turns


def make_diarizer(name: str, hf_token: str) -> Diarizer:
    """`name` で diarizer を解決する。"""
    if name == "pyannote":
        return PyannoteDiarizer(hf_token)
    if name == "sortformer":
        return SortformerDiarizer()
    raise ValueError(f"未知の diarizer: {name} (使用可能: pyannote, sortformer)")


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


# 後方互換 (Task 1 後の API): cli.py が `from speaks.diarize import diarize` を使うため
def diarize(audio_path: str, hf_token: str) -> SpeakerTurns:
    return PyannoteDiarizer(hf_token)(audio_path)
