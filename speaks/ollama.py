"""Ollama HTTP クライアントの薄いラッパ。"""

import httpx

from speaks.io import OLLAMA_BASE_URL


def generate(model: str, prompt: str, timeout: float = 300.0) -> str:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]
