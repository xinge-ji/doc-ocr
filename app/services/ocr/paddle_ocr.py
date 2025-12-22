from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

from app.core.config import settings
from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.ocr.base import BaseOcrClient
from app.services.ocr.preprocess import OcrPreprocessor

logger = logging.getLogger(__name__)


class PaddleOcrClient(BaseOcrClient):
    """用 PaddleOCR det+rec 跑 OCR，生成结构化结果并落地可视化."""

    def __init__(
        self,
        *,
        ocr: PaddleOCR | None = None,
        visualizer: Callable[..., Any] | None = None,
        save_visualization: bool = True,
        vis_dir: str | Path | None = None,
        preprocessor: OcrPreprocessor | None = None,
    ) -> None:
        self._ocr = ocr or self._build_ocr()
        self._visualizer = visualizer or draw_ocr
        self.save_visualization = save_visualization
        self.vis_dir = Path(vis_dir or settings.paddle_ocr_vis_dir)
        self.preprocessor = preprocessor or OcrPreprocessor()

    def _build_ocr(self) -> PaddleOCR:
        logger.info("--- INIT PaddleOCR(det+rec) ---")
        return PaddleOCR(
            text_detection_model_name=settings.paddle_det_model_name,
            text_recognition_model_name=settings.paddle_rec_model_name,
            use_doc_orientation_classify=settings.paddle_use_doc_orientation_classify,
            use_doc_unwarping=settings.paddle_use_doc_unwarping,
            use_textline_orientation=settings.paddle_use_textline_orientation,
            device=settings.paddle_ocr_device,
        )

    async def extract(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> OcrResult:
        run_id = uuid.uuid4().hex
        today = datetime.now().strftime("%Y%m%d")

        try:
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
                raw_results = await loop.run_in_executor(
                    None, lambda p=str(page_info.path): self._ocr.ocr(p, cls=False)
                )
                if not raw_results:
                    continue

                page = raw_results[0]
                page_items = self._parse_page(page, page_info.page)
                items.extend(page_items)

                if self.save_visualization and page_items:
                    try:
                        self._save_visualization(
                            page_info.path,
                            page,
                            today=today,
                            run_id=run_id,
                            page_idx=page_info.page,
                        )
                    except Exception as exc:  # pragma: no cover - 可视化失败不阻塞主流程
                        logger.warning("保存可视化失败: %s", exc)

            logger.info("PaddleOCR 提取完成，items=%d", len(items))
            return OcrResult(items=items)

    def _save_visualization(
        self,
        image_path: Path,
        page: Sequence[Any],
        *,
        today: str,
        run_id: str,
        page_idx: int,
    ) -> None:
        image = Image.open(image_path).convert("RGB")
        boxes = [line[0] for line in page if self._valid_line(line)]
        texts = [line[1][0] for line in page if self._valid_line(line)]
        scores = [line[1][1] for line in page if self._valid_line(line)]

        if not boxes:
            return

        vis_array = self._visualizer(image, boxes, texts, scores)
        vis_img = Image.fromarray(vis_array)
        target_path = self._build_vis_path(today=today, run_id=run_id, page_idx=page_idx)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        vis_img.save(target_path)

    def _build_vis_path(self, *, today: str, run_id: str, page_idx: int) -> Path:
        filename = f"page{page_idx}.jpg"
        return self.vis_dir / today / run_id / filename

    def _parse_page(self, page: Sequence[Any], page_idx: int) -> list[OcrItem]:
        items: list[OcrItem] = []
        for line in page:
            if not self._valid_line(line):
                continue
            box_points = line[0]
            text_val, _score = line[1]
            bbox = self._to_bounding_box(box_points)
            items.append(OcrItem(text=str(text_val), bounding_box=bbox, page=page_idx))
        return items

    def _to_bounding_box(self, box_points: Sequence[Sequence[float]]) -> BoundingBox:
        # PaddleOCR box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        p0, _, p2, _ = box_points
        return BoundingBox(x1=float(p0[0]), y1=float(p0[1]), x2=float(p2[0]), y2=float(p2[1]))

    def _valid_line(self, line: Any) -> bool:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            return False
        box, text_info = line[0], line[1]
        if not self._valid_box(box):
            return False
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
            return False
        if not text_info[0]:
            return False
        return True

    def _valid_box(self, box: Any) -> bool:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        return all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box)
