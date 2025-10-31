from __future__ import annotations

from typing import Iterable
from ..config import settings

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer. "
    "If the answer is not in the context, say you don't know."
)


class LLMProvider:
    def __init__(self):
        self.provider = settings.llm_provider
        self.openai_model = settings.openai_model
        self._client = None
        # OpenAI (native)
        if self.provider == "openai" and settings.openai_api_key and OpenAI is not None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        # OpenRouter via OpenAI SDK
        elif self.provider == "openrouter" and settings.openrouter_key and OpenAI is not None:
            default_headers = {}
            if settings.app_public_url:
                default_headers["HTTP-Referer"] = settings.app_public_url
            if settings.app_title:
                default_headers["X-Title"] = settings.app_title
            self._client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_key,
                default_headers=default_headers or None,
            )

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self._client:
            model = (
                settings.openrouter_model
                if self.provider == "openrouter"
                else self.openai_model
            )
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        # Local naive fallback: extract sentences containing query keywords
        user_last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        keywords = {w.lower() for w in user_last.split() if len(w) > 3}
        context = " ".join(m["content"] for m in messages if m["role"] == "system" and "Context:" in m["content"]) 
        sentences = context.split(". ")
        selected = [s for s in sentences if any(k in s.lower() for k in keywords)]
        if not selected:
            return "I don't know from the provided context."
        return ". ".join(selected)[:1200]

