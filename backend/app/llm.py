"""Optional abstractive summarization via an OpenAI-compatible chat API."""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-4o-mini"

_LENGTH_hint = {
    "short": "in 2-3 sentences",
    "medium": "in one concise paragraph (4-6 sentences)",
    "detailed": "as a thorough multi-paragraph summary covering all key points",
}


def llm_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def abstractive_summarize(text: str, length: str = "medium") -> str:
    """Call an OpenAI-compatible chat completions endpoint. Raises on any failure."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    hint = _length_hint.get(length, _length_hint["medium"])
    system = (
        "You are a precise document summarizer. Summarize the user's document "
        f"{hint}. Return only the summary, no preamble."
    )
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text[:12000]},
            ],
            "temperature": 0.2,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError("Empty completion from LLM")
    return content.strip()
