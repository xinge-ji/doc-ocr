from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Sequence

from paddleocr import PaddleOCRVL

from app.core.config import settings
from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.ocr.base import BaseOcrClient

logger = logging.getLogger(__name__)


class PaddleVLOcrClient(BaseOcrClient):
    """OCR client wrapper around PaddleOCRVL using the configured vLLM backend."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        backend: str | None = None,
        server_url: str | None = None,
        layout_model_name: str | None = None,
        pipeline: PaddleOCRVL | None = None,
    ) -> None:
        """Initialize the PaddleOCRVL pipeline with configured defaults."""

        self.model_dir = Path(model_dir or settings.doclayout_model_path)
        self.backend = backend or settings.ocr_vl_rec_backend
        self.server_url = server_url or settings.ocr_vl_rec_server_url
        self.layout_model_name = layout_model_name or settings.ocr_layout_model_name
        self._pipeline = pipeline or self._build_pipeline()

    def _build_pipeline(self) -> PaddleOCRVL:
        """Construct the PaddleOCRVL pipeline using vLLM server backend."""

        logger.info(
            "Initializing PaddleOCRVL pipeline (backend=%s, server=%s, layout_model=%s)",
            self.backend,
            self.server_url,
            self.layout_model_name,
        )
        return PaddleOCRVL(
            vl_rec_backend=self.backend,
            vl_rec_server_url=self.server_url,
            layout_detection_model_name=self.layout_model_name,
            layout_detection_model_dir=str(self.model_dir),
        )

    async def extract(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> OcrResult:
        """Run OCR on a path/URL or raw bytes and return normalized OCR results."""

        path, cleanup = self._prepare_source(
            source,
            filename=filename,
            content_type=content_type,
        )
        try:
            loop = asyncio.get_running_loop()
            raw_output = await loop.run_in_executor(None, lambda: self._pipeline.predict(path))
        except Exception as exc:  # pragma: no cover - passthrough for service errors
            raise RuntimeError("PaddleOCRVL prediction failed") from exc
        finally:
            cleanup()

        items = self._to_ocr_items(raw_output)
        return OcrResult(items=items)

    def _prepare_source(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, Callable[[], None]]:
        """Normalize input into a filesystem path PaddleOCRVL can consume."""

        if isinstance(source, str):
            return source, lambda: None

        suffix = self._infer_suffix(filename=filename, content_type=content_type)
        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(source)
        temp_file.flush()
        temp_file.close()

        def _cleanup() -> None:
            Path(temp_file.name).unlink(missing_ok=True)

        return temp_file.name, _cleanup

    def _infer_suffix(self, *, filename: str | None, content_type: str | None) -> str:
        """Pick an extension acceptable by PaddleOCRVL."""

        if filename:
            suffix = Path(filename).suffix
            if suffix:
                return suffix

        content_type_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/bmp": ".bmp",
            "application/pdf": ".pdf",
        }
        if content_type:
            mapped = content_type_map.get(content_type.lower())
            if mapped:
                return mapped

        return ".jpg"

    def _to_ocr_items(self, raw_output: Sequence[Any]) -> list[OcrItem]:
        """Convert PaddleOCRVL prediction output into OcrItem list."""

        items: list[OcrItem] = []
        for page_idx, block in self._iter_blocks(raw_output):
            text = str(block.get("block_content", "")).strip()
            bbox = block.get("block_bbox")
            if not text or not self._valid_bbox(bbox):
                continue

            items.append(
                OcrItem(
                    text=text,
                    bounding_box=self._to_bounding_box(bbox),
                    page=page_idx + 1,
                    block_id=block.get("block_id"),
                    line_id=block.get("block_order"),
                ),
            )
        return items

    def _iter_blocks(self, raw_output: Sequence[Any]) -> Iterable[tuple[int, dict[str, Any]]]:
        """Yield (page_index, block_dict) tuples from PaddleOCRVL output."""

        for page_idx, page in enumerate(raw_output):
            parsing_res_list = getattr(page, "parsing_res_list", None)
            if parsing_res_list is None and isinstance(page, dict):
                parsing_res_list = page.get("parsing_res_list")
            if not parsing_res_list:
                continue

            for block in parsing_res_list:
                if isinstance(block, dict):
                    yield page_idx, block

    def _valid_bbox(self, bbox: Any) -> bool:
        """Validate bbox shape matches [x1, y1, x2, y2]."""

        return isinstance(bbox, (list, tuple)) and len(bbox) == 4

    def _to_bounding_box(self, bbox: Sequence[float]) -> BoundingBox:
        """Convert PaddleOCRVL rectangular bbox to a compact XYXY bounding box."""

        x1, y1, x2, y2 = bbox
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
