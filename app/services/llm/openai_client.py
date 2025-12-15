from __future__ import annotations

import json
import logging
import random
from typing import Any

from openai import AsyncOpenAI

from app.core.config import LlmNode, settings
from app.services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client built on the OpenAI SDK for structured JSON responses."""

    def __init__(self, nodes: list[LlmNode] | None = None) -> None:
        """Create async OpenAI clients for configured LLM nodes."""

        self._nodes = nodes or settings.llm_nodes
        self._clients: dict[str, AsyncOpenAI] = {}

        for node in self._nodes:
            base_url = self._normalize_base_url(node.base_url)
            self._clients[node.name] = AsyncOpenAI(
                api_key=node.api_key,
                base_url=base_url,
            )

    async def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Invoke chat.completions on a selected node and return parsed content."""

        node_name: str | None = kwargs.pop("node_name", None)
        node = self._pick_node(node_name)
        client = self._clients[node.name]

        model = kwargs.pop("model", node.model)
        system_prompt = kwargs.pop("system_prompt", None)
        messages = kwargs.pop("messages", None)

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        response_format = kwargs.pop("response_format", {"type": "json_object"})

        logger.debug("Calling LLM node=%s model=%s with %d messages", node.name, model, len(messages))
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )

        content = completion.choices[0].message.content
        logger.debug("LLM node=%s return: %s", node.name, content)
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

    def _pick_node(self, node_name: str | None) -> LlmNode:
        """Pick a node by name or randomly when unspecified."""

        if node_name:
            key = node_name.strip().lower()
            for node in self._nodes:
                if node.name == key:
                    return node
            raise ValueError(f"LLM node not found: {node_name}")

        return random.choice(self._nodes)

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
