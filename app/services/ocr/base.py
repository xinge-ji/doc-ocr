from typing import Protocol, runtime_checkable

from app.schemas.ocr import OcrResult


@runtime_checkable
class BaseOcrClient(Protocol):
    async def extract(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> OcrResult:
        """Run OCR on the provided source and return structured results."""
        ...
