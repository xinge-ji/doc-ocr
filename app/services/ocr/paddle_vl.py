from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Sequence

from paddleocr import PaddleOCRVL

from app.core.config import settings
from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.ocr.base import BaseOcrClient

# 即使 logger 不打印，print 肯定会打印
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
        self.model_dir = Path(model_dir or settings.doclayout_model_path)
        self.backend = backend or settings.ocr_vl_rec_backend
        self.server_url = server_url or settings.ocr_vl_rec_server_url
        self.layout_model_name = layout_model_name or settings.ocr_layout_model_name
        self._pipeline = pipeline or self._build_pipeline()

    def _build_pipeline(self) -> PaddleOCRVL:
        print(f"--- INIT PADDLE: {self.server_url} ---")
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
        # 1. 准备文件
        path, cleanup = self._prepare_source(
            source,
            filename=filename,
            content_type=content_type,
        )

        # DEBUG: 打印文件路径和大小，确保文件存在且不为空
        file_size = os.path.getsize(path)
        print(f"DEBUG: Processing temp file: {path}, Size: {file_size} bytes")

        try:
            loop = asyncio.get_running_loop()
            # 执行预测
            raw_output = await loop.run_in_executor(None, lambda: self._pipeline.predict(path))

            # --- 核心调试区 ---
            print(f"DEBUG: Raw Output Type: {type(raw_output)}")

            # 尝试把结果保存下来，就像你的脚本里做的一样
            try:
                if raw_output and hasattr(raw_output, "__iter__"):
                    for i, res in enumerate(raw_output):
                        # 尝试保存到项目根目录
                        save_path = f"debug_api_output_{i}.json"
                        if hasattr(res, "save_to_json"):
                            res.save_to_json(save_path)
                            print(f"DEBUG: Saved result to {os.path.abspath(save_path)}")
                        else:
                            print(f"DEBUG: Result item {i} does not have save_to_json: {res}")
            except Exception as e:
                print(f"DEBUG: Failed to save debug json: {e}")
            # ----------------

        except Exception as exc:
            print(f"ERROR: Paddle execution failed: {exc}")
            raise RuntimeError("PaddleOCRVL prediction failed") from exc
        finally:
            cleanup()

        items = self._to_ocr_items(raw_output)
        print(f"DEBUG: Parsed items count: {len(items)}")
        return OcrResult(items=items)

    def _prepare_source(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, Callable[[], None]]:
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
        if filename:
            suffix = Path(filename).suffix
            if suffix:
                return suffix
        content_type_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
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

        # 如果 raw_output 是 None 或空
        if not raw_output:
            print("DEBUG: raw_output is empty or None")
            return []

        for page_idx, page in enumerate(raw_output):
            # PaddleOCRVL 的 output 元素通常是一个对象，内部存了 parsing_res_list
            # 我们先打印看看这个 page 到底有什么属性
            # print(f"DEBUG: Inspecting Page {page_idx}: {dir(page)}")

            parsing_res_list = getattr(page, "parsing_res_list", None)
            if parsing_res_list is None and isinstance(page, dict):
                parsing_res_list = page.get("parsing_res_list")

            if not parsing_res_list:
                print(f"DEBUG: Page {page_idx} has no parsing_res_list")
                continue

            for block in parsing_res_list:
                # 兼容 block 可能是对象也可能是 dict
                if isinstance(block, dict):
                    text = str(block.get("block_content", "")).strip()
                    bbox = block.get("block_bbox")
                    block_id = block.get("block_id")
                    line_id = block.get("block_order")
                else:
                    # 假如是对象
                    text = str(getattr(block, "block_content", "")).strip()
                    bbox = getattr(block, "block_bbox", None)
                    block_id = getattr(block, "block_id", None)
                    line_id = getattr(block, "block_order", None)

                if not text or not self._valid_bbox(bbox):
                    continue

                items.append(
                    OcrItem(
                        text=text,
                        bounding_box=self._to_bounding_box(bbox),
                        page=page_idx + 1,
                        block_id=block_id,
                        line_id=line_id,
                    )
                )
        return items

    def _iter_blocks(self, raw_output: Sequence[Any]) -> Iterable[tuple[int, dict[str, Any]]]:
        # 这个方法暂时弃用，逻辑已合并到 _to_ocr_items 以方便调试
        pass

    def _valid_bbox(self, bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)):
            return False
        # 简单检查
        return len(bbox) in (4, 8)

    def _to_bounding_box(self, bbox: Sequence[Any]) -> BoundingBox:
        x1, y1, x2, y2 = self._normalize_bbox(bbox)
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _normalize_bbox(self, bbox: Sequence[Any]) -> tuple[float, float, float, float]:
        if len(bbox) == 4 and all(isinstance(val, (int, float)) for val in bbox):
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2), float(y2)
        if len(bbox) == 8:
            xs = bbox[0::2]
            ys = bbox[1::2]
            return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        if len(bbox) == 4:
            xs = [float(pt[0]) for pt in bbox]
            ys = [float(pt[1]) for pt in bbox]
            return min(xs), min(ys), max(xs), max(ys)
        return 0.0, 0.0, 0.0, 0.0
