"""Ollama を使った議事録要約。"""

import sys
import time

from speaks import ollama
from speaks.prompts import SUMMARY_PROMPT


def summarize(transcript: str, model: str, step: str = "[3/3]") -> str:
    """Ollama で議事録風に要約"""
    print(f"{step} 要約中... (model: {model})", file=sys.stderr)
    start = time.time()

    result = ollama.generate(model, SUMMARY_PROMPT.format(transcript=transcript))

    elapsed = time.time() - start
    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)
    return result
