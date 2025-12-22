from __future__ import annotations

import json
import logging
import time
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
        llm_node: str | None = None,
    ) -> InvoiceData:
        """Execute OCR then LLM to produce structured invoice data."""

        start_time = time.perf_counter()
        logger.info(f"🚀 [TIMER] Pipeline started for file: {filename}")

        # --- Phase 1: OCR ---
        t0 = time.perf_counter()
        logger.info("🕒 [TIMER] Step 1: Starting OCR extraction...")

        ocr_result = await self.ocr_client.extract(
            source,
            filename=filename,
            content_type=content_type,
        )
        logger.debug(f"OCR result: {ocr_result}")

        t1 = time.perf_counter()
        ocr_duration = t1 - t0
        logger.info(f"✅ [TIMER] Step 1: OCR finished. Duration: {ocr_duration:.4f}s")
        # --------------------

        # --- Phase 2: Build Prompt ---
        system_prompt, user_prompt = self._build_prompts(ocr_result)

        # --- Phase 3: LLM ---
        t2 = time.perf_counter()
        logger.info("🕒 [TIMER] Step 2: Sending request to LLM...")

        llm_payload = await self.llm_client.generate_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0,
            node_name=llm_node,
        )

        t3 = time.perf_counter()
        llm_duration = t3 - t2
        logger.info(f"✅ [TIMER] Step 2: LLM finished. Duration: {llm_duration:.4f}s")
        # --------------------

        total_duration = time.perf_counter() - start_time
        logger.info(
            f"🏁 [TIMER] Pipeline completed. Total: {total_duration:.4f}s (OCR: {ocr_duration:.2f}s, LLM: {llm_duration:.2f}s)"
        )

        try:
            return InvoiceData.model_validate(llm_payload)
        except ValidationError as exc:  # pragma: no cover - runtime validation guard
            logger.error(f"❌ [TIMER] Validation failed after {total_duration:.2f}s")
            raise ValueError("LLM response failed invoice schema validation") from exc

    def _build_prompts(self, ocr_result: OcrResult) -> tuple[str, str]:
        """Render system and user prompts with OCR context."""

        if not ocr_result.items:
            logger.warning("OCR returned no items; LLM will receive empty OCR payload")

        ocr_json = json.dumps(
            self._serialize_ocr_items(ocr_result.items),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user_prompt = self.user_prompt_template.format(ocr_json=ocr_json)
        return self.system_prompt, user_prompt

    def _serialize_ocr_items(self, items: list[OcrItem]) -> list[dict[str, Any]]:
        """Convert OCR items into a LLM-friendly JSON structure."""

        ordered = sorted(items, key=lambda i: i.page)
        serialized: list[dict[str, Any]] = []
        for idx, item in enumerate(ordered):
            serialized.append(
                {
                    "text": item.text,
                    "bbox": item.bounding_box.as_xyxy(),
                    "page": item.page,
                    "idx": idx,
                }
            )
        return serialized
