"""Whisper segments と pyannote 話者ターンを word 単位でマージする。"""

from speaks.diarize import SpeakerTurns, find_speaker


def merge_words_by_speaker(segments, speaker_turns: SpeakerTurns | None) -> str:
    """Whisperのsegment境界を尊重しつつ、segment内では連続する同一話者の発話をマージする。"""
    lines: list[str] = []
    for seg in segments:
        if speaker_turns is None:
            lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()}")
            continue

        words = seg.words or []
        if not words:
            mid = (seg.start + seg.end) / 2
            speaker = find_speaker(mid, speaker_turns)
            lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {speaker}: {seg.text.strip()}")
            continue

        groups: list[list] = []  # [[speaker, [word,...], start, end], ...]
        for w in words:
            wmid = (w.start + w.end) / 2
            speaker = find_speaker(wmid, speaker_turns)
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(w.word)
                groups[-1][3] = w.end
            else:
                groups.append([speaker, [w.word], w.start, w.end])

        for speaker, ws, start, end in groups:
            lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {''.join(ws).strip()}")

    return "\n".join(lines)
