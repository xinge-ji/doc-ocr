from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Iterable

import numpy as np
from paddleocr import DocPreprocessor

logger = logging.getLogger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None
    logger.warning("OpenCV not available; basic image enhancement disabled")


class OcrPreprocessor:
    """前处理封装：旋转、去扭曲、基础增强。"""

    def __init__(
        self,
        *,
        use_doc_orientation: bool = True,
        use_doc_unwarping: bool = True,
        use_basic_enhance: bool = True,
    ) -> None:
        self.use_doc_orientation = use_doc_orientation
        self.use_doc_unwarping = use_doc_unwarping
        self.use_basic_enhance = use_basic_enhance and cv2 is not None

        self._docpp = None
        if self.use_doc_orientation or self.use_doc_unwarping:
            self._docpp = DocPreprocessor(
                use_doc_orientation_classify=self.use_doc_orientation,
                use_doc_unwarping=self.use_doc_unwarping,
            )

    def process(self, path: str) -> tuple[list[str], Callable[[], None]]:
        """运行预处理，返回处理后文件路径列表与清理函数。"""

        cleanup_actions: list[Callable[[], None]] = []
        current_paths = [path]

        # 先跑文档预处理（旋转/去扭曲）
        if self._docpp is not None:
            docpp_paths, docpp_cleanup = self._run_doc_preprocessor(current_paths)
            cleanup_actions.append(docpp_cleanup)
            current_paths = docpp_paths

        # 再做基础增强
        if self.use_basic_enhance:
            enhanced_paths, enhance_cleanup = self._run_basic_enhance(current_paths)
            cleanup_actions.append(enhance_cleanup)
            current_paths = enhanced_paths

        def _cleanup() -> None:
            for fn in reversed(cleanup_actions):
                try:
                    fn()
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning(f"Cleanup failed: {exc}")

        return current_paths, _cleanup

    def _run_doc_preprocessor(self, paths: Iterable[str]) -> tuple[list[str], Callable[[], None]]:
        temp_dir = TemporaryDirectory(prefix="ocr_docpp_")
        try:
            results = self._docpp.predict(list(paths))
            for res in results:
                res.save_to_img(temp_dir.name)

            saved_files = sorted(Path(temp_dir.name).glob("*"))
            if not saved_files:
                return list(paths), temp_dir.cleanup

            return [str(p) for p in saved_files], temp_dir.cleanup
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(f"DocPreprocessor failed, fallback to raw image: {exc}")
            temp_dir.cleanup()
            return list(paths), lambda: None

    def _run_basic_enhance(self, paths: Iterable[str]) -> tuple[list[str], Callable[[], None]]:
        if cv2 is None:
            return list(paths), lambda: None

        temp_files: list[str] = []

        def _cleanup() -> None:
            for file_path in temp_files:
                Path(file_path).unlink(missing_ok=True)

        processed_paths: list[str] = []
        try:
            for path in paths:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    processed_paths.append(path)
                    continue

                processed = self._enhance_image(image)
                suffix = Path(path).suffix or ".png"
                temp_file = NamedTemporaryFile(delete=False, suffix=suffix, prefix="ocr_enh_")

                success, encoded = cv2.imencode(suffix, processed)
                if not success:
                    Path(temp_file.name).unlink(missing_ok=True)
                    processed_paths.append(path)
                    continue

                encoded.tofile(temp_file.name)
                temp_file.flush()
                temp_file.close()

                temp_files.append(temp_file.name)
                processed_paths.append(temp_file.name)

            return processed_paths, _cleanup
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(f"Basic enhance failed, fallback to raw image: {exc}")
            _cleanup()
            return list(paths), lambda: None

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """简单增强：轻降噪 + CLAHE + 自适应阈值。"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blur)

        thresh = cv2.adaptiveThreshold(
            clahe_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
