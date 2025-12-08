from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client built on the OpenAI SDK for structured JSON responses."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        """Create an async OpenAI client using environment-driven configuration."""

        base_url = self._normalize_base_url(settings.llm_base_url)
        self._client = client or AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=base_url,
        )

    async def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Invoke chat.completions with JSON-mode and return parsed content."""

        model = kwargs.pop("model", settings.llm_model_name)
        system_prompt = kwargs.pop("system_prompt", None)
        messages = kwargs.pop("messages", None)

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        response_format = kwargs.pop("response_format", {"type": "json_object"})

        logger.debug(f"Calling LLM model={model} with {messages} messages")
        completion = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )

        content = completion.choices[0].message.content
        logger.debug(f"LLM return: {content}")
        if not content:
            raise ValueError("LLM returned empty content")

        # 移除可能存在的 Markdown 代码块标记
        cleaned_content = content.strip()
        if cleaned_content.startswith("```"):
            import re
            cleaned_content = re.sub(r"^```(json)?|```$", "", cleaned_content, flags=re.MULTILINE | re.DOTALL).strip()
    
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("LLM response is not valid JSON") from exc

    def _normalize_base_url(self, base_url: str) -> str:
        """Avoid duplicate /chat/completions suffixes when users pass full endpoints."""

        normalized = base_url.rstrip("/")
        lowered = normalized.lower()
        for suffix in ("/chat/completions", "/v1/chat/completions", "/v4/chat/completions"):
            if lowered.endswith(suffix):
                trimmed = normalized[: -len(suffix)]
                logger.warning("Stripped trailing %s from llm_base_url", suffix)
                return trimmed
        return normalized
