"""音声前処理: 16kHz / mono / ピーク正規化した WAV を返す。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

from speaks.io import TARGET_SR


def preprocess(audio_path: Path, out_dir: Path) -> Path:
    """16kHz mono + ピーク正規化で前処理した WAV を返す。"""
    print("[0/N] 前処理中 (16kHz/mono/正規化)...", file=sys.stderr)
    start = time.time()

    sr, data = wavfile.read(str(audio_path))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio).unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    peak = wav.abs().max()
    if peak > 0:
        wav = wav * (0.95 / peak)

    out_path = out_dir / f"{audio_path.stem}_16k.wav"
    wav_int16 = (wav.squeeze(0).numpy() * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), TARGET_SR, wav_int16)

    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒 → {out_path})", file=sys.stderr)
    return out_path
