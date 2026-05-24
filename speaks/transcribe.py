"""faster-whisper による日本語 ASR。"""

import sys
import time

from faster_whisper import WhisperModel

from speaks.diarize import SpeakerTurns
from speaks.merge import merge_words_by_speaker


def transcribe(
    audio_path: str,
    model_size: str,
    speaker_turns: SpeakerTurns | None = None,
    initial_prompt: str | None = None,
) -> str:
    """faster-whisper で音声を文字起こし (VAD・幻覚抑制・word timestamps 有効)"""
    step = "[2/3]" if speaker_turns is not None else "[1/2]"
    print(f"{step} 文字起こし中... (model: {model_size})", file=sys.stderr)
    start = time.time()

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )
    segments = list(segments)

    elapsed = time.time() - start
    print(
        f"    完了 ({elapsed:.1f}秒, 言語: {info.language}, 確率: {info.language_probability:.0%})",
        file=sys.stderr,
    )

    return merge_words_by_speaker(segments, speaker_turns)
