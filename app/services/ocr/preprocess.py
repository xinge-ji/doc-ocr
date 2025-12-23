"""OCR 预处理：表格线抑制 + 透视校正 + 方向旋转。"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import cv2
import numpy as np
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
        label = self._extract_orientation_label(res)
        angle = self._label_to_angle(label)
        self._log_orientation_debug(res, angle)
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

        warped = self._apply_perspective(image)
        rotated = self._apply_rotation(warped, angle)
        logger.debug(
            "旋转落盘: path=%s angle=%d size=%sx%s",
            image_path,
            angle,
            rotated.shape[1],
            rotated.shape[0],
        )
        target_path = self._build_output_path(image_path, angle, page, today=today, run_id=run_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target_path), rotated):
            raise RuntimeError(f"写文件失败：{target_path}")
        return target_path

    def _apply_perspective(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        table_mask = self._detect_table_lines(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 2, 2)
        edges = cv2.Canny(blurred, 60, 240, apertureSize=3)
        if table_mask is not None:
            edges = cv2.bitwise_and(edges, cv2.bitwise_not(table_mask))
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contour = self._find_max_contour(edges)
        if contour is None:
            logger.debug("透视跳过: 未找到可用轮廓")
            return image

        box = self._get_box_points(contour)
        if box is None:
            logger.debug("透视跳过: 未提取到四点")
            return image

        box = self._order_points(box)
        if not self._is_perspective_confident(box, gray.shape):
            logger.debug("透视跳过: 外框可信度不足")
            return image

        width = self._point_distance(box[0], box[1])
        height = self._point_distance(box[1], box[2])
        if width <= 0 or height <= 0:
            logger.debug("透视跳过: 目标尺寸异常")
            return image

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(box, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    def _detect_table_lines(self, gray: np.ndarray) -> np.ndarray | None:
        height, width = gray.shape[:2]
        min_side = min(height, width)
        if min_side < 3:
            return None

        block_size = min(15, min_side if min_side % 2 == 1 else min_side - 1)
        block_size = max(3, block_size)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            10,
        )

        horizontal_size = min(max(15, width // 20), width)
        vertical_size = min(max(15, height // 20), height)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        table_mask = cv2.bitwise_or(horizontal, vertical)

        margin_x = min(max(5, int(width * 0.03)), width // 2)
        margin_y = min(max(5, int(height * 0.03)), height // 2)
        if margin_y > 0:
            table_mask[:margin_y, :] = 0
            table_mask[-margin_y:, :] = 0
        if margin_x > 0:
            table_mask[:, :margin_x] = 0
            table_mask[:, -margin_x:] = 0

        return table_mask

    def _apply_rotation(self, image, angle: int):
        if angle % 360 == 0:
            return image
        normalized = angle % 360
        if normalized == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if normalized == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # 非标准角度走仿射
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), normalized, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))

    def _find_max_contour(self, edges: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) <= 0:
            return None
        return max_contour

    def _get_box_points(self, contour: np.ndarray) -> np.ndarray | None:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        if box is None or len(box) != 4:
            return None
        return box.astype("float32")

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _point_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sqrt(np.sum(np.square(a - b))))

    def _is_perspective_confident(self, box: np.ndarray, image_shape: tuple[int, int]) -> bool:
        height, width = image_shape
        if height <= 0 or width <= 0:
            return False

        contour = box.reshape(4, 1, 2).astype("float32")
        corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        hits = 0
        for corner in corners:
            if cv2.pointPolygonTest(contour, (float(corner[0]), float(corner[1])), False) >= 0:
                hits += 1
        return hits >= 2

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

    def _extract_orientation_label(self, res: Any) -> str:
        payload = self._unwrap_orientation_payload(res)
        label_names = self._read_orientation_value(payload, "label_names") or []
        label = label_names[0] if label_names else "0"
        return str(label)

    def _log_orientation_debug(self, res: Any, angle: int) -> None:
        payload = self._unwrap_orientation_payload(res)
        scores = self._format_orientation_value(self._read_orientation_value(payload, "scores"))
        if scores is not None:
            logger.info("方向预测 scores=%s", scores)

        logger.debug(
            "方向模型输出: input_path=%s page_index=%s class_ids=%s label_names=%s angle=%s",
            self._read_orientation_value(payload, "input_path"),
            self._read_orientation_value(payload, "page_index"),
            self._format_orientation_value(self._read_orientation_value(payload, "class_ids")),
            self._read_orientation_value(payload, "label_names"),
            angle,
        )

    def _unwrap_orientation_payload(self, res: Any) -> Any:
        if isinstance(res, dict):
            return res.get("res", res)
        inner = getattr(res, "res", None)
        return inner if inner is not None else res

    def _read_orientation_value(self, payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    def _format_orientation_value(self, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

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

