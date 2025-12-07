from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import ValidationError

from app.prompts import load_prompt
from app.schemas.invoice import InvoiceData
from app.schemas.ocr import OcrItem, OcrResult
from app.services.llm.base import BaseLLMClient
from app.services.ocr.base import BaseOcrClient

logger = logging.getLogger(__name__)


class InvoiceExtractionPipeline:
    """Hybrid pipeline orchestrating OCR and LLM invoice extraction."""

    def __init__(
        self,
        ocr_client: BaseOcrClient,
        llm_client: BaseLLMClient,
        *,
        prompt_loader: Callable[[str], str] = load_prompt,
    ) -> None:
        self.ocr_client = ocr_client
        self.llm_client = llm_client
        self._prompt_loader = prompt_loader
        self.system_prompt = prompt_loader("invoice_extraction_system")
        self.user_prompt_template = prompt_loader("invoice_extraction_user")

    async def run(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> InvoiceData:
        """Execute OCR then LLM to produce structured invoice data."""

        ocr_result = await self.ocr_client.extract(
            source,
            filename=filename,
            content_type=content_type,
        )
        system_prompt, user_prompt = self._build_prompts(ocr_result)

        llm_payload = await self.llm_client.generate_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0,
        )
        try:
            return InvoiceData.model_validate(llm_payload)
        except ValidationError as exc:  # pragma: no cover - runtime validation guard
            raise ValueError("LLM response failed invoice schema validation") from exc

    def _build_prompts(self, ocr_result: OcrResult) -> tuple[str, str]:
        """Render system and user prompts with OCR context."""

        if not ocr_result.items:
            logger.warning("OCR returned no items; LLM will receive empty OCR payload")

        ocr_json = json.dumps(
            self._serialize_ocr_items(ocr_result.items),
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = self.user_prompt_template.format(ocr_json=ocr_json)
        return self.system_prompt, user_prompt

    def _serialize_ocr_items(self, items: list[OcrItem]) -> list[dict[str, Any]]:
        """Convert OCR items into a LLM-friendly JSON structure."""

        return [
            {
                "text": item.text,
                "bbox": item.bounding_box.as_xyxy(),
                "page": item.page,
            }
            for item in sorted(
                items,
                key=lambda i: (i.page, i.line_id or 0, i.block_id or 0),
            )
        ]
