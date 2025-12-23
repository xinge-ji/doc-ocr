"""Usage: invoice extraction pipeline (OCR -> rules)."""

from __future__ import annotations

import logging
import time

from app.schemas.ocr import OcrResult
from app.services.ocr.base import BaseOcrClient
from app.services.rules.invoice_rule_extractor import InvoiceRuleExtractor, RuleExtractionResult

logger = logging.getLogger(__name__)


class InvoiceExtractionPipeline:
    """Pipeline orchestrating OCR and rule-based invoice extraction."""

    def __init__(
        self,
        ocr_client: BaseOcrClient,
        *,
        rule_extractor: InvoiceRuleExtractor | None = None,
    ) -> None:
        self.ocr_client = ocr_client
        self.rule_extractor = rule_extractor or InvoiceRuleExtractor()

    async def run(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> RuleExtractionResult:
        """Execute OCR then rules to produce template-specific invoice data."""

        start_time = time.perf_counter()
        logger.info("?? [TIMER] Pipeline started for file: %s", filename)

        t0 = time.perf_counter()
        logger.info("?? [TIMER] Step 1: Starting OCR extraction...")

        ocr_result = await self.ocr_client.extract(
            source,
            filename=filename,
            content_type=content_type,
        )
        logger.debug("OCR result: %s", ocr_result)

        t1 = time.perf_counter()
        ocr_duration = t1 - t0
        logger.info("? [TIMER] Step 1: OCR finished. Duration: %.4fs", ocr_duration)

        rule_result = self._run_rules(ocr_result)
        if not rule_result.complete:
            logger.warning(
                "Rule extraction failed: template=%s errors=%s",
                rule_result.template_name,
                rule_result.errors,
            )

        total_duration = time.perf_counter() - start_time
        logger.info(
            "?? [TIMER] Pipeline completed. Total: %.4fs (OCR: %.2fs)",
            total_duration,
            ocr_duration,
        )
        return rule_result

    def _run_rules(self, ocr_result: OcrResult) -> RuleExtractionResult:
        return self.rule_extractor.extract(ocr_result)
