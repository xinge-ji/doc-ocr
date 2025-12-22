from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import httpx

from app.core.config import settings
from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.ocr.base import BaseOcrClient
from app.services.ocr.preprocess import OcrPreprocessor

logger = logging.getLogger(__name__)


class HunyuanOcrClient(BaseOcrClient):
    """调用 Hunyuan OCR (OpenAI 兼容接口) 并转成通用 OCR 结构."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        preprocessor: OcrPreprocessor | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = self._normalize_base_url(base_url or settings.hunyuan_base_url)
        self.api_key = api_key or settings.hunyuan_api_key
        self.model = model or settings.hunyuan_model
        self.preprocessor = preprocessor or OcrPreprocessor()
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def extract(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> OcrResult:
        loop = asyncio.get_running_loop()
        items: list[OcrItem] = []

        preprocess_result = await loop.run_in_executor(
            None,
            lambda: self.preprocessor.preprocess(
                source,
                filename=filename,
                content_type=content_type,
            ),
        )

        if not preprocess_result.pages:
            logger.info("预处理返回空页面：%s", filename or "bytes")
            return OcrResult(items=[])

        for page_info in preprocess_result.pages:
            payload = self._build_payload(page_info.path)
            logger.debug("Hunyuan payload for page %s: %s", page_info.page, payload)

            response_json = await self._post(payload)
            content = self._extract_content(response_json)

            page_items = self._parse_content(content, page=page_info.page)
            items.extend(page_items)

        logger.info("Hunyuan OCR 提取完成，items=%d", len(items))
        return OcrResult(items=items)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _build_payload(self, image_path: Path) -> dict[str, Any]:
        data_url = self._to_data_url(image_path)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": "检测并识别图片中的文字，将文本坐标格式化输出。",
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "top_k": 1,
            "repetition_penalty": 1.0,
        }

    def _extract_content(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError("Hunyuan OCR 返回空 choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise ValueError("Hunyuan OCR 返回空 content")
        return str(content)

    def _parse_content(self, content: str, *, page: int) -> list[OcrItem]:
        items: list[OcrItem] = []
        pattern = re.compile(
            r"(?P<prefix>.*?)\(\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\)"
            r"\s*,\s*\(\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\)",
            re.DOTALL,
        )

        search_pos = 0
        while True:
            match = pattern.search(content, search_pos)
            if not match:
                break
            text = match.group("prefix").strip()
            bbox = self._bbox_from_match(match)
            if text and bbox:
                items.append(
                    OcrItem(
                        text=text,
                        bounding_box=bbox,
                        page=page,
                        block_id=None,
                        line_id=None,
                    )
                )
            search_pos = match.end()

        return items

    def _bbox_from_match(self, match: re.Match[str]) -> BoundingBox | None:
        try:
            x1 = float(match.group("x1"))
            y1 = float(match.group("y1"))
            x2 = float(match.group("x2"))
            y2 = float(match.group("y2"))
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except (TypeError, ValueError):
            return None

    def _normalize_base_url(self, base_url: str) -> str:
        lowered = base_url.rstrip("/")
        suffixes: Sequence[str] = ("/chat/completions", "/v1/chat/completions")
        for suffix in suffixes:
            if lowered.lower().endswith(suffix):
                return lowered[: -len(suffix)]
        return lowered

    def _to_data_url(self, image_path: Path) -> str:
        data = image_path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        mime = "image/jpeg"
        suffix = image_path.suffix.lower()
        if suffix in {".png"}:
            mime = "image/png"
        return f"data:{mime};base64,{encoded}"
