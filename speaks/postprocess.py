"""幻覚抑制の後処理: 低信頼除去 / 反復縮約 / 定型句除外。

論文の知見:
- Barański+ 2025 (arXiv:2501.11378): 非発話区間で Whisper が定型句を出すため、
  avg_logprob / no_speech_prob と定型句辞書で後処理することが有効。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# faster_whisper.transcribe.Segment は属性ベースなので、テスト用に同じ形のダミーを
# 受け取れるよう duck typing で扱う。
LOW_CONF_AVG_LOGPROB_MAX = -1.5
LOW_CONF_NO_SPEECH_PROB_MIN = 0.5
REPEAT_THRESHOLD = 3  # 同一 token 列が N 回以上連続したら 1 回に縮約


@dataclass
class FilterStats:
    dropped_low_conf: int = 0
    dropped_boilerplate: int = 0
    collapsed_repeats: int = 0
    dropped_phrases: list[str] = field(default_factory=list)


def load_hallucination_phrases(path: Path | None) -> list[str]:
    """`#` コメント / 空行をスキップして定型句リストを返す。`None` なら空リスト。"""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"hallucination phrases ファイルが見つかりません: {path}")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def default_phrases_path() -> Path:
    """同梱の日本語デフォルト辞書のパス。"""
    return Path(__file__).parent / "data" / "hallucination_phrases.ja.txt"


def _is_low_confidence(seg) -> bool:
    avg_logprob = getattr(seg, "avg_logprob", None)
    no_speech_prob = getattr(seg, "no_speech_prob", None)
    if avg_logprob is None or no_speech_prob is None:
        return False
    return avg_logprob < LOW_CONF_AVG_LOGPROB_MAX and no_speech_prob > LOW_CONF_NO_SPEECH_PROB_MIN


def _collapse_repeated_words(words: list, stats: FilterStats):
    """同じ token (word.word の strip) が REPEAT_THRESHOLD 回以上連続したら 1 回に縮約。

    word は faster_whisper の Word 相当 (word, start, end を持つオブジェクト) を想定。
    """
    if not words or len(words) < REPEAT_THRESHOLD:
        return words

    out = []
    run_token: str | None = None
    run_count = 0
    run_first = None
    for w in words:
        token = (w.word or "").strip()
        if token and token == run_token:
            run_count += 1
            continue
        # flush 前の run
        if run_first is not None:
            if run_count >= REPEAT_THRESHOLD:
                stats.collapsed_repeats += 1
                out.append(run_first)  # 1 回だけ保持
            else:
                out.extend([run_first] * run_count)
        run_token = token
        run_count = 1
        run_first = w
    # 末尾
    if run_first is not None:
        if run_count >= REPEAT_THRESHOLD:
            stats.collapsed_repeats += 1
            out.append(run_first)
        else:
            out.extend([run_first] * run_count)
    return out


def filter_segments(segments, phrases: list[str], enable: bool = True) -> tuple[list, FilterStats]:
    """segments を後処理して幻覚を抑える。

    enable=False のときは何もせず返す。
    """
    stats = FilterStats()
    if not enable:
        return list(segments), stats

    phrase_set = {p.strip() for p in phrases if p.strip()}
    out = []
    for seg in segments:
        text = (seg.text or "").strip()

        if _is_low_confidence(seg):
            stats.dropped_low_conf += 1
            continue

        if text in phrase_set:
            stats.dropped_boilerplate += 1
            stats.dropped_phrases.append(text)
            continue

        # word 列が取れるなら反復縮約
        words = getattr(seg, "words", None) or []
        if words:
            new_words = _collapse_repeated_words(list(words), stats)
            # 縮約後に text を再合成して seg を新オブジェクトに差し替える
            if len(new_words) != len(words):
                new_text = "".join((w.word or "") for w in new_words).strip()
                seg = _replace_segment(seg, words=new_words, text=new_text)

        out.append(seg)
    return out, stats


def _replace_segment(seg, *, words, text):
    """seg を words/text 差し替えた軽量プロキシで返す (Segment が frozen な場合に備えて)。"""
    try:
        # dataclasses.replace は faster_whisper の Segment (namedtuple-like) でも動くことがある
        from dataclasses import is_dataclass, replace

        if is_dataclass(seg):
            return replace(seg, words=words, text=text)
    except Exception:
        pass

    class _Proxy:
        pass

    proxy = _Proxy()
    for attr in ("start", "end", "avg_logprob", "no_speech_prob", "id", "seek", "tokens", "temperature", "compression_ratio"):
        if hasattr(seg, attr):
            setattr(proxy, attr, getattr(seg, attr))
    proxy.words = words
    proxy.text = text
    return proxy


def print_stats(stats: FilterStats) -> None:
    if (
        stats.dropped_low_conf
        or stats.dropped_boilerplate
        or stats.collapsed_repeats
    ):
        print(
            f"[postprocess] 低信頼除去={stats.dropped_low_conf}, "
            f"定型句除去={stats.dropped_boilerplate}, "
            f"反復縮約={stats.collapsed_repeats}",
            file=sys.stderr,
        )
