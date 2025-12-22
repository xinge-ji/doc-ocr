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
            raw_results = await loop.run_in_executor(
                None, lambda p=str(page_info.path): self._ocr.predict(p)
            )
            if not raw_results:
                continue

            for result in raw_results:
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

    def _save_visualization_from_result(
        self, result: Any, *, today: str, run_id: str, page_idx: int
    ) -> None:
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

        json_data = self._get_json_data(result)
        if isinstance(json_data, dict):
            boxes = self._first_seq(json_data, ["boxes", "dt_polys", "polygons", "points"])
            texts = self._first_seq(json_data, ["rec_text", "text", "texts"])
            scores = self._first_seq(json_data, ["rec_score", "score", "scores"])
            entries.extend(self._zip_entries(boxes, texts, scores))

        ocr_res = getattr(result, "ocr_res", None)
        if isinstance(ocr_res, list):
            entries.extend(self._extract_from_ocr_res(ocr_res))

        boxes_attr = getattr(result, "boxes", None)
        texts_attr = getattr(result, "boxes_txt", None) or getattr(result, "txts", None)
        scores_attr = getattr(result, "boxes_score", None) or getattr(result, "scores", None)
        entries.extend(self._zip_entries(boxes_attr, texts_attr, scores_attr))

        return entries

    def _get_json_data(self, result: Any) -> Any:
        json_data = getattr(result, "json", None)
        if callable(json_data):
            try:
                return json_data()
            except TypeError:
                return None
        return json_data

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

    def _first_seq(self, data: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = data.get(key)
            if isinstance(value, (list, tuple)):
                return value
        return None

    def _extract_from_ocr_res(self, ocr_res: Sequence[Any]) -> list[tuple[Sequence[Any], str, float | None]]:
        entries: list[tuple[Sequence[Any], str, float | None]] = []
        for line in ocr_res:
            if not self._valid_line(line):
                continue
            box_points = line[0]
            text_info = line[1]
            text_val = self._text_from_entry(text_info)
            score_val = None
            if isinstance(text_info, (list, tuple)) and len(text_info) > 1:
                try:
                    score_val = float(text_info[1])
                except (TypeError, ValueError):
                    score_val = None
            entries.append((box_points, text_val, score_val))
        return entries

    def _text_from_entry(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        value_str = str(value).strip()
        return value_str or None

    def _to_bounding_box(self, box_points: Any) -> BoundingBox | None:
        # PaddleOCR box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if not self._valid_box(box_points):
            return None
        points = [self._coerce_point(pt) for pt in box_points]
        if len(points) != 4 or any(p is None for p in points):
            return None
        p0, _, p2, _ = points
        return BoundingBox(x1=p0[0], y1=p0[1], x2=p2[0], y2=p2[1])

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
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            return False
        return all(self._valid_point(pt) for pt in box)

    def _valid_point(self, pt: Any) -> bool:
        if not isinstance(pt, Sequence) or isinstance(pt, (str, bytes)) or len(pt) != 2:
            return False
        return True

    def _coerce_point(self, pt: Any) -> tuple[float, float] | None:
        if not self._valid_point(pt):
            return None
        try:
            return float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            return None

