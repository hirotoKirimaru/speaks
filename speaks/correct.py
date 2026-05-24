"""LLM (Ollama) による ASR 補正。

行数 / タイムスタンプ / 話者ラベルの構造を保持したまま、表記揺れと同音異義のみを
LLM に直してもらう。行数不一致や明らかに壊れた出力が来た場合は fail-open
(補正なし) で進める。
"""

import sys
import time

from speaks import ollama
from speaks.prompts import CORRECTION_PROMPT


def correct_transcript(
    transcript: str,
    model: str,
    vocab: list[str] | None = None,
    step: str = "[2/3]",
) -> str:
    """ASR 出力を LLM で補正する。

    行数が変わったり明らかに失敗した場合は元の transcript を返す (fail-open)。
    """
    print(f"{step} ASR 補正中... (model: {model})", file=sys.stderr)
    start = time.time()

    vocab_section = "(指定なし)" if not vocab else "、".join(vocab)
    prompt = CORRECTION_PROMPT.format(vocab=vocab_section, transcript=transcript)

    try:
        result = ollama.generate(model, prompt)
    except Exception as e:
        print(f"    エラー: {e} → 補正なしで進めます", file=sys.stderr)
        return transcript

    elapsed = time.time() - start

    in_lines = transcript.splitlines()
    out_lines = result.strip().splitlines()

    # コードブロック開始/終端を取り除く救済 (LLM が ``` で囲ってきた場合)
    out_lines = [line for line in out_lines if not line.strip().startswith("```")]

    if len(out_lines) != len(in_lines):
        print(
            f"    警告: 行数不一致 (入力 {len(in_lines)} → 出力 {len(out_lines)})。補正破棄。",
            file=sys.stderr,
        )
        return transcript

    print(f"    完了 ({elapsed:.1f}秒)", file=sys.stderr)
    return "\n".join(out_lines)
