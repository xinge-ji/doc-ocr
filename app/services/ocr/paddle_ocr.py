from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from paddleocr import PaddleOCR
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
        save_visualization: bool = True,
        vis_dir: str | Path | None = None,
        preprocessor: OcrPreprocessor | None = None,
    ) -> None:
        self._ocr = ocr or self._build_ocr()
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
            paddlex_config="PaddleOCR.yaml",
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
            raw_results = await loop.run_in_executor(None, lambda p=str(page_info.path): self._ocr.predict(p))
            if not raw_results:
                continue

            for result in raw_results:
                self._debug_result_meta(result)
                if self.save_visualization:
                    try:
                        self._save_visualization_from_result(
                            result,
                            today=today,
                            run_id=run_id,
                            page_idx=page_info.page,
                        )
                    except Exception as exc:  # pragma: no cover - 可视化失败不阻塞主流程
                        logger.warning("保存可视化失败: %s", exc)

                page_items = self._parse_predict_result(result, page_info.page)
                items.extend(page_items)

        logger.info("PaddleOCR 提取完成，items=%d", len(items))
        return OcrResult(items=items)

    def _save_visualization_from_result(self, result: Any, *, today: str, run_id: str, page_idx: int) -> None:
        vis_dir = self._build_vis_dir(today=today, run_id=run_id)
        vis_dir.mkdir(parents=True, exist_ok=True)

        img_dict = getattr(result, "img", None)
        if isinstance(img_dict, dict):
            vis_img = img_dict.get("ocr_res_img")
            if isinstance(vis_img, Image.Image):
                target_path = vis_dir / f"page{page_idx}.jpg"
                vis_img.save(target_path)
                return

        saver = getattr(result, "save_to_img", None)
        if callable(saver):
            saver(save_path=str(vis_dir))
            return

        logger.warning("可视化结果不可用，缺少 img/save_to_img 属性")

    def _build_vis_dir(self, *, today: str, run_id: str) -> Path:
        return self.vis_dir / today / run_id

    def _parse_predict_result(self, result: Any, page_idx: int) -> list[OcrItem]:
        items: list[OcrItem] = []

        for box_points, text_val, _score in self._iter_result_entries(result):
            bbox = self._to_bounding_box(box_points)
            if bbox is None:
                continue
            items.append(OcrItem(text=str(text_val), bounding_box=bbox, page=page_idx))

        return items

    def _iter_result_entries(self, result: Any) -> list[tuple[Sequence[Any], str, float | None]]:
        entries: list[tuple[Sequence[Any], str, float | None]] = []

        boxes_attr = self._get_from_result(result, "rec_boxes")
        texts_attr = self._get_from_result(result, "rec_texts")
        scores_attr = self._get_from_result(result, "rec_scores")
        entries.extend(self._zip_entries(boxes_attr, texts_attr, scores_attr))

        return entries

    def _debug_result_meta(self, result: Any) -> None:
        try:
            texts = self._get_from_result(result, "rec_texts") or []
            boxes = self._get_from_result(result, "rec_boxes")
            scores = self._get_from_result(result, "rec_scores") or []
            logger.debug(
                "PaddleOCR rec_texts len=%s boxes type=%s scores len=%s",
                len(texts),
                type(boxes),
                len(scores),
            )
        except Exception:
            pass

    def _get_from_result(self, result: Any, key: str) -> Any:
        if isinstance(result, dict):
            return result.get(key)
        if hasattr(result, "get"):
            try:
                return result.get(key)
            except Exception:
                pass
        return getattr(result, key, None)

    def _zip_entries(
        self,
        boxes: Any,
        texts: Any,
        scores: Any = None,
    ) -> list[tuple[Sequence[Any], str, float | None]]:
        if not isinstance(boxes, (list, tuple)) or not isinstance(texts, (list, tuple)):
            return []

        length = min(len(boxes), len(texts))
        entries: list[tuple[Sequence[Any], str, float | None]] = []

        for idx in range(length):
            text_val = self._text_from_entry(texts[idx])
            if text_val is None:
                continue

            score_val = None
            if isinstance(scores, (list, tuple)) and idx < len(scores):
                try:
                    score_val = float(scores[idx])
                except (TypeError, ValueError):
                    score_val = None

            entries.append((boxes[idx], text_val, score_val))

        return entries

    def _text_from_entry(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        value_str = str(value).strip()
        return value_str or None

    def _to_bounding_box(self, box_points: Any) -> BoundingBox | None:
        # PaddleOCR rec_boxes is expected as [x1, y1, x2, y2]
        if not isinstance(box_points, (list, tuple)) or len(box_points) != 4:
            return None
        try:
            x1, y1, x2, y2 = box_points
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except Exception:
            return None
