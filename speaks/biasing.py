"""語彙バイアシング (Phase 1): vocab ファイル読み込みと initial_prompt 構築。"""

import sys
from pathlib import Path

VOCAB_PROMPT_PREFIX = "用語: "
DEFAULT_PROMPT_MAX_CHARS = 100


def load_vocab(path: Path) -> list[str]:
    """vocab ファイルから 1 行 1 語の語彙リストを読み込む。

    `#` で始まる行と空行は無視する。
    """
    if not path.exists():
        raise FileNotFoundError(f"vocab ファイルが見つかりません: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    vocab: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vocab.append(line)
    return vocab


def build_initial_prompt(
    user_prompt: str | None,
    vocab: list[str],
    max_chars: int = DEFAULT_PROMPT_MAX_CHARS,
) -> str | None:
    """ユーザ指定の initial_prompt と vocab を結合して Whisper に渡す文字列を作る。

    Whisper の prompt は内部的に 224 token 上限がある。日本語では 1 文字 ≒ 2 token
    程度になりがちなので、デフォルト 100 文字で切り詰める。

    切り詰めが発生したら stderr に警告を出す。
    """
    parts: list[str] = []
    if user_prompt:
        parts.append(user_prompt.strip())

    if vocab:
        kept: list[str] = []
        used = sum(len(p) for p in parts) + len(VOCAB_PROMPT_PREFIX)
        for word in vocab:
            # 1 単語 + 区切り `、`
            cost = len(word) + 1
            if used + cost > max_chars:
                dropped = len(vocab) - len(kept)
                print(
                    f"[biasing] prompt 上限 ({max_chars} 文字) に収まらず {dropped} 語を切り詰めました",
                    file=sys.stderr,
                )
                break
            kept.append(word)
            used += cost
        if kept:
            parts.append(VOCAB_PROMPT_PREFIX + "、".join(kept))

    if not parts:
        return None
    return " ".join(parts)


def warn_unimplemented_biasing(mode: str) -> str:
    """biasing-mode が trie / both のとき、Phase 1 では prompt にフォールバックする。"""
    if mode in ("trie", "both"):
        print(
            f"[biasing] --biasing-mode {mode} は未実装です。prompt モードで動作します。",
            file=sys.stderr,
        )
        return "prompt"
    return mode
