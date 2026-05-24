"""wav2vec2 ベースの forced alignment。

Whisper 内蔵 word_timestamps より細粒度かつ正確な word 区間を得る。
WhisperX (Bain+ 2023, arXiv:2303.00747) の forced alignment 思想を、
依存を増やさず最小限の Viterbi で実装する。

各 segment ごとに wav2vec2 + CTC ロジットを取り、Whisper segment の text を
character/token 系列として整列させる。失敗時はその segment を元のまま残す。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

DEFAULT_ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    probability: float | None = None


class Wav2Vec2Aligner:
    """wav2vec2 + CTC で character 単位の forced alignment を行う。"""

    def __init__(self, model_id: str = DEFAULT_ALIGN_MODEL, device: str | None = None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[align] wav2vec2 アライナをロード中... ({model_id})", file=sys.stderr)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def align_segment(
        self, audio: np.ndarray, sr: int, text: str
    ) -> list[AlignedWord] | None:
        """単一 segment の音声 (mono float32) と text を受け、word 区間を返す。

        失敗した場合は None。
        """
        if sr != self.sampling_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio_t)
            audio = audio_t.squeeze(0).numpy()

        # text を空白で分割した「word」を整列対象にする。日本語は連続文字列なので、
        # ここでは text 全体を 1 word として扱うフォールバックを用意し、最終的には
        # character ベースの alignment 結果を Whisper の word 列にマージする。
        if not text.strip():
            return None

        try:
            inputs = self.processor(
                audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits[0]  # (T, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            tokenizer = self.processor.tokenizer
            blank_id = tokenizer.pad_token_id

            # text を character 列に展開
            chars = list(text)
            token_ids: list[int] = []
            for c in chars:
                ids = tokenizer.encode(c, add_special_tokens=False)
                if not ids:
                    # 辞書外文字。整列不可。
                    return None
                token_ids.append(ids[0])

            trellis = _build_trellis(log_probs.cpu(), token_ids, blank_id)
            path = _backtrack(trellis, log_probs.cpu(), token_ids, blank_id)
            if path is None:
                return None
            segments_per_char = _merge_path_to_chars(path, chars)

            # frame → time
            frame_dur = len(audio) / self.sampling_rate / log_probs.shape[0]
            words: list[AlignedWord] = []
            for ch_start, ch_end, ch in segments_per_char:
                words.append(
                    AlignedWord(
                        word=ch,
                        start=ch_start * frame_dur,
                        end=ch_end * frame_dur,
                    )
                )
            return words
        except Exception as e:
            print(f"[align] segment のアライメント失敗: {e}", file=sys.stderr)
            return None


def _build_trellis(log_probs, token_ids, blank_id):
    num_frames = log_probs.shape[0]
    num_tokens = len(token_ids)
    trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
    trellis[0, 0] = 0.0
    for t in range(num_frames):
        trellis[t + 1, 0] = trellis[t, 0] + log_probs[t, blank_id]
        for j in range(1, num_tokens + 1):
            stay = trellis[t, j] + log_probs[t, blank_id]
            change = trellis[t, j - 1] + log_probs[t, token_ids[j - 1]]
            trellis[t + 1, j] = max(stay, change)
    return trellis


def _backtrack(trellis, log_probs, token_ids, blank_id):
    j = trellis.shape[1] - 1
    t = trellis.shape[0] - 1
    if torch.isinf(trellis[t, j]).item():
        return None
    path = []
    while t > 0 and j >= 0:
        stay = trellis[t - 1, j] + log_probs[t - 1, blank_id]
        change = (
            trellis[t - 1, j - 1] + log_probs[t - 1, token_ids[j - 1]]
            if j > 0
            else -float("inf")
        )
        if change > stay and j > 0:
            path.append((t - 1, j - 1, False))  # token emit
            j -= 1
        else:
            path.append((t - 1, j, True))  # blank/stay
        t -= 1
    path.reverse()
    return path


def _merge_path_to_chars(path, chars):
    """path から各 char の (start_frame, end_frame, char) を作る。"""
    if not chars:
        return []
    # path: list of (frame, token_idx, is_blank)
    # token_idx は 0..len(chars). token_idx=k は chars[k-1] を emit している区間
    spans = []
    current_idx = None
    current_start = None
    last_frame = 0
    for frame, j, is_blank in path:
        if not is_blank:
            # token を emit: chars[j] (0-based) が対応
            if current_idx == j:
                pass
            else:
                if current_idx is not None and current_idx < len(chars):
                    spans.append((current_start, frame, chars[current_idx]))
                current_idx = j
                current_start = frame
        last_frame = frame
    if current_idx is not None and current_idx < len(chars):
        spans.append((current_start, last_frame + 1, chars[current_idx]))
    return spans


def realign_segments(
    aligner: Wav2Vec2Aligner,
    segments,
    wav_path: str,
):
    """各 segment の word 区間を wav2vec2 で取り直す。

    segment 単位で失敗した場合はその segment は元のまま残す。
    """
    if not segments:
        return list(segments)

    print("[align] forced alignment 実行中...", file=sys.stderr)
    start = time.time()
    sr, full = wavfile.read(wav_path)
    if full.dtype == np.int16:
        full_audio = full.astype(np.float32) / 32768.0
    elif full.dtype == np.int32:
        full_audio = full.astype(np.float32) / 2147483648.0
    elif full.dtype == np.uint8:
        full_audio = (full.astype(np.float32) - 128.0) / 128.0
    else:
        full_audio = full.astype(np.float32)
    if full_audio.ndim == 2:
        full_audio = full_audio.mean(axis=1)

    aligned_segs = []
    failures = 0
    for seg in segments:
        seg_start = max(0, int(seg.start * sr))
        seg_end = min(len(full_audio), int(seg.end * sr))
        chunk = full_audio[seg_start:seg_end]
        if len(chunk) < sr * 0.05:
            aligned_segs.append(seg)
            continue
        aligned_chars = aligner.align_segment(chunk, sr, seg.text or "")
        if not aligned_chars:
            failures += 1
            aligned_segs.append(seg)
            continue
        new_words = _aligned_chars_to_words(aligned_chars, seg.text or "", seg.start)
        if not new_words:
            failures += 1
            aligned_segs.append(seg)
            continue
        aligned_segs.append(_replace_seg_words(seg, new_words))

    elapsed = time.time() - start
    print(
        f"    完了 ({elapsed:.1f}秒, アライン失敗 segment={failures})",
        file=sys.stderr,
    )
    return aligned_segs


def _aligned_chars_to_words(chars: list[AlignedWord], text: str, segment_start: float):
    """character 単位の AlignedWord 列を、Whisper の word 風オブジェクト列に変換する。

    日本語では「word」=「複数文字」相当のため、空白で分割して word を作り直す。
    空白が無い場合は文全体を 1 word として扱う。
    """
    if not chars:
        return None

    # 元 text を空白で word に分割し、各 word の文字数で chars を groupping
    word_strs = text.split()
    if not word_strs:
        word_strs = [text]
    word_strs = [w for w in word_strs if w]
    if not word_strs:
        return None

    out = []
    ci = 0
    total_chars = sum(len(w) for w in word_strs)
    if total_chars > len(chars):
        # 整列不一致 (辞書外文字でスキップされた場合など)
        return None
    for w in word_strs:
        if ci >= len(chars):
            return None
        start = chars[ci].start + segment_start
        end_idx = min(ci + len(w) - 1, len(chars) - 1)
        end = chars[end_idx].end + segment_start
        out.append(_WordLike(word=w, start=start, end=end, probability=None))
        ci += len(w)
    return out


class _WordLike:
    """faster_whisper.Word と互換 (word, start, end, probability)。"""

    __slots__ = ("word", "start", "end", "probability")

    def __init__(self, word: str, start: float, end: float, probability: float | None):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


def _replace_seg_words(seg: Any, new_words):
    class _Proxy:
        pass

    proxy = _Proxy()
    for attr in (
        "start",
        "end",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "id",
        "seek",
        "tokens",
        "temperature",
        "compression_ratio",
    ):
        if hasattr(seg, attr):
            setattr(proxy, attr, getattr(seg, attr))
    proxy.words = new_words
    return proxy
