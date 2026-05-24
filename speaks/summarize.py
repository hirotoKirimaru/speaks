"""Ollama を使った議事録要約。単段 / 多段パイプライン両対応。"""

import sys
import time

from speaks import ollama
from speaks.prompts import MINUTES_PROMPT, SUMMARY_PROMPT, TOPIC_PROMPT


def summarize(
    transcript: str,
    model: str,
    step: str = "[3/3]",
) -> str:
    """単段プロンプト (--summary-pipeline single)。既存挙動。"""
    print(f"{step} 要約中... (model: {model})", file=sys.stderr)
    start = time.time()
    result = ollama.generate(model, SUMMARY_PROMPT.format(transcript=transcript))
    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)
    return result


def summarize_multi(
    transcript: str,
    model: str,
    step: str = "[3/3]",
) -> tuple[str, str]:
    """多段プロンプト: (1) トピック抽出 → (2) 議事録合成。

    返り値: (topics_md, minutes_md)
    """
    print(f"{step} トピック抽出中... (model: {model})", file=sys.stderr)
    start = time.time()
    topics = ollama.generate(model, TOPIC_PROMPT.format(transcript=transcript)).strip()
    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)

    print(f"{step} 議事録合成中... (model: {model})", file=sys.stderr)
    start = time.time()
    minutes = ollama.generate(
        model, MINUTES_PROMPT.format(topics=topics, transcript=transcript)
    )
    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)

    return topics, minutes


def run_summary(
    transcript: str,
    model: str,
    mode: str,
    step: str = "[3/3]",
) -> dict[str, str]:
    """`mode` (single|multi) に応じて要約を実行する。

    返り値の dict は cli から保存される: `minutes` 必須、`topics` は multi のみ。
    """
    if mode == "single":
        return {"minutes": summarize(transcript, model, step)}
    if mode == "multi":
        topics, minutes = summarize_multi(transcript, model, step)
        return {"topics": topics, "minutes": minutes}
    raise ValueError(f"未知の summary-pipeline: {mode} (使用可能: single, multi)")
