from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Sequence

from paddleocr import PaddleOCRVL

from app.core.config import settings
from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.ocr.base import BaseOcrClient

# 使用 print 确保能看到日志
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
        path, cleanup = self._prepare_source(
            source,
            filename=filename,
            content_type=content_type,
        )
        print(f"DEBUG: Processing temp file: {path}")

        try:
            loop = asyncio.get_running_loop()
            raw_output = await loop.run_in_executor(None, lambda: self._pipeline.predict(path))

            # --- 简化版：直接解析 ---
            # PaddleOCRVL.predict 返回的是一个 list，里面每个元素对应一页
            # 每个元素可能是一个 dict (你给的JSON情况) 或者一个对象

            items = []
            if raw_output:
                for page_idx, page_data in enumerate(raw_output):
                    # 尝试保存调试文件
                    try:
                        save_path = f"debug_api_output_{page_idx}.json"
                        if hasattr(page_data, "save_to_json"):
                            page_data.save_to_json(save_path)
                            print(f"DEBUG: Saved result to {os.path.abspath(save_path)}")
                    except Exception:
                        pass

                    # 提取 items
                    page_items = self._parse_single_page(page_data, page_idx + 1)
                    items.extend(page_items)

            print(f"DEBUG: Total parsed items: {len(items)}")
            if len(items) == 0:
                print("WARNING: Parsed 0 items! Check parsing logic.")

            return OcrResult(items=items)

        except Exception as exc:
            print(f"ERROR: Paddle execution failed: {exc}")
            raise RuntimeError("PaddleOCRVL prediction failed") from exc
        finally:
            cleanup()

    def _parse_single_page(self, page_data: Any, page_num: int) -> list[OcrItem]:
        """专门适配 PaddleOCRVL 输出结构的解析器"""
        items = []

        # 1. 获取 parsing_res_list
        # 情况A: page_data 是 dict (你的情况)
        if isinstance(page_data, dict):
            res_list = page_data.get("parsing_res_list")
        # 情况B: page_data 是对象
        else:
            res_list = getattr(page_data, "parsing_res_list", None)

        if not res_list:
            print(f"DEBUG: Page {page_num} has no parsing_res_list")
            return []

        # 2. 遍历 block
        for block in res_list:
            # 兼容 block 是 dict 或 对象
            if isinstance(block, dict):
                label = block.get("block_label", "")
                text = str(block.get("block_content", "")).strip()
                bbox = block.get("block_bbox")
                block_id = block.get("block_id")
            else:
                label = getattr(block, "block_label", "")
                text = str(getattr(block, "block_content", "")).strip()
                bbox = getattr(block, "block_bbox", None)
                block_id = getattr(block, "block_id", None)

            # 过滤逻辑：
            # - 忽略 image, seal (通常是空的或者乱码)
            # - 保留 table (里面有 HTML)
            # - 保留 text, paragraph_title 等
            if label in ("image", "seal"):
                continue

            if not text:
                continue

            if not self._valid_bbox(bbox):
                continue

            # 构造 OcrItem
            items.append(
                OcrItem(
                    text=text,
                    bounding_box=self._to_bounding_box(bbox),
                    page=page_num,
                    block_id=block_id,
                    # label 也可以存，但 OcrItem 目前没定义，暂时忽略
                )
            )

        return items

    def _prepare_source(
        self,
        source: str | bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, Callable[[], None]]:
        if isinstance(source, str):
            return source, lambda: None

        # 你的图片是 .jpg，这里逻辑没问题
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

        return temp_file.name, _cleanup

    def _valid_bbox(self, bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)):
            return False
        # [x1, y1, x2, y2]
        return len(bbox) == 4

    def _to_bounding_box(self, bbox: Sequence[Any]) -> BoundingBox:
        # 你的 JSON 里 bbox 就是 [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
