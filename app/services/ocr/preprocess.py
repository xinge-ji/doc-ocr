from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import cv2
from paddleocr import DocImgOrientationClassification

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessPageResult:
    path: Path
    angle: int
    page: int


@dataclass
class PreprocessResult:
    pages: list[PreprocessPageResult]

    @property
    def paths(self) -> list[Path]:
        return [p.path for p in self.pages]


class OcrPreprocessor:
    """这个预处理把图片/pdf先纠偏再落盘给 OCR 用。"""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        orientation_model: DocImgOrientationClassification | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or settings.preprocess_output_dir)
        self._orientation_model = orientation_model

    def preprocess(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> PreprocessResult:
        """
        输入路径或二进制，自动方向识别并旋转，返回落盘后的文件路径列表。
        PDF 会先按页转图片再处理。
        """
        temp_path, cleanup = self._ensure_local_path(source, filename=filename, content_type=content_type)
        today = datetime.now().strftime("%Y%m%d")
        run_id = uuid.uuid4().hex

        try:
            page_images = self._explode_to_images(temp_path, content_type=content_type)
            rotated_pages: list[PreprocessPageResult] = []
            for page_idx, img_path in enumerate(page_images, start=1):
                angle = self._detect_angle(img_path)
                rotated_path = self._rotate_and_save(
                    img_path,
                    angle,
                    page_idx,
                    today=today,
                    run_id=run_id,
                )
                rotated_pages.append(PreprocessPageResult(path=rotated_path, angle=angle, page=page_idx))

            logger.info("预处理完成，输出 %d 页", len(rotated_pages))
            return PreprocessResult(pages=rotated_pages)
        finally:
            cleanup()

    def _ensure_local_path(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> tuple[Path, Callable[[], None]]:
        if isinstance(source, str):
            return Path(source), lambda: None

        suffix = ".jpg"
        if filename and Path(filename).suffix:
            suffix = Path(filename).suffix
        elif content_type == "application/pdf":
            suffix = ".pdf"

        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(source)
        temp_file.flush()
        temp_file.close()

        def _cleanup() -> None:
            Path(temp_file.name).unlink(missing_ok=True)

        return Path(temp_file.name), _cleanup

    def _explode_to_images(self, path: Path, *, content_type: str | None) -> list[Path]:
        # PDF 走按页转图
        if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
            try:
                from paddleocr.tools.infer.utility import pdf2img  # type: ignore
            except Exception as exc:  # pragma: no cover - 避免环境差异
                logger.error("缺少 pdf2img 工具，无法处理 PDF: %s", exc)
                raise RuntimeError("pdf2img not available for PDF preprocessing") from exc

            images = pdf2img(str(path))
            if not images:
                raise RuntimeError("PDF 转图片失败，未得到任何页面")
            return [Path(img_path) for img_path in images]

        # 非 PDF 直接当作单图
        return [path]

    def _detect_angle(self, image_path: Path) -> int:
        model = self._get_orientation_model()
        output = model.predict(str(image_path), batch_size=1)
        if not output:
            logger.warning("方向分类未返回结果，默认 0 度：%s", image_path)
            return 0

        res = output[0]
        labels = getattr(res, "label_names", None) or []
        label = labels[0] if labels else "0"
        angle = self._label_to_angle(label)
        logger.info("方向预测 %s -> %d 度", label, angle)
        return angle

    def _rotate_and_save(
        self,
        image_path: Path,
        angle: int,
        page: int,
        *,
        today: str,
        run_id: str,
    ) -> Path:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"读图失败：{image_path}")

        rotated = self._apply_rotation(image, angle)
        target_path = self._build_output_path(image_path, angle, page, today=today, run_id=run_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target_path), rotated):
            raise RuntimeError(f"写文件失败：{target_path}")
        return target_path

    def _apply_rotation(self, image, angle: int):
        if angle % 360 == 0:
            return image
        normalized = angle % 360
        if normalized == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if normalized == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 非标准角度走仿射
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), normalized, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))

    def _build_output_path(
        self,
        image_path: Path,
        angle: int,
        page: int,
        *,
        today: str,
        run_id: str,
    ) -> Path:
        ext = image_path.suffix or ".jpg"
        filename = f"page{page}_rot{angle}{ext}"
        return self.output_dir / today / run_id / filename

    def _get_orientation_model(self) -> DocImgOrientationClassification:
        if self._orientation_model is None:
            self._orientation_model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        return self._orientation_model

    def _label_to_angle(self, label: str) -> int:
        label_norm = str(label).lower()
        mapping = {
            "0": 0,
            "0_degree": 0,
            "90": 90,
            "90_degree": 90,
            "180": 180,
            "180_degree": 180,
            "270": 270,
            "270_degree": 270,
        }
        return mapping.get(label_norm, 0)
