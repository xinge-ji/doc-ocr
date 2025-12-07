from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Sequence

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

            items = []
            if raw_output:
                for page_idx, page_obj in enumerate(raw_output):
                    # --- 核心修改：回读策略 ---
                    # 既然 page_obj.save_to_json() 能生成正确的文件，我们就利用它
                    # 创建一个临时的 json 文件路径
                    json_temp_name = f"{path}_res_{page_idx}.json"

                    try:
                        # 1. 保存到磁盘
                        if hasattr(page_obj, "save_to_json"):
                            page_obj.save_to_json(json_temp_name)

                            # 2. 读回来转成 Dict
                            with open(json_temp_name, "r", encoding="utf-8") as f:
                                page_dict = json.load(f)

                            print(f"DEBUG: Successfully re-loaded JSON for page {page_idx}")

                            # 3. 解析 Dict
                            page_items = self._parse_single_page_dict(page_dict, page_idx + 1)
                            items.extend(page_items)
                        else:
                            print(f"WARNING: Page object {page_idx} has no save_to_json method")

                    except Exception as e:
                        print(f"ERROR processing page {page_idx}: {e}")
                    finally:
                        # 清理这个临时的 JSON 文件
                        if os.path.exists(json_temp_name):
                            os.remove(json_temp_name)

            print(f"DEBUG: Total parsed items: {len(items)}")
            return OcrResult(items=items)

        except Exception as exc:
            print(f"ERROR: Paddle execution failed: {exc}")
            raise RuntimeError("PaddleOCRVL prediction failed") from exc
        finally:
            cleanup()

    def _parse_single_page_dict(self, page_data: dict, page_num: int) -> list[OcrItem]:
        """解析标准的 Dict 结构"""
        items = []
        res_list = page_data.get("parsing_res_list", [])

        if not res_list:
            print(f"DEBUG: Page {page_num} has empty parsing_res_list")
            return []

        for block in res_list:
            label = block.get("block_label", "")
            # 注意：Paddle 有时候返回 null，转成字符串如果是 'None' 需要处理
            content_val = block.get("block_content")
            text = str(content_val).strip() if content_val is not None else ""
            bbox = block.get("block_bbox")
            block_id = block.get("block_id")

            # 过滤逻辑
            if label in ("image", "seal"):
                continue

            if not text:
                continue

            if not self._valid_bbox(bbox):
                continue

            items.append(
                OcrItem(
                    text=text,
                    bounding_box=self._to_bounding_box(bbox),
                    page=page_num,
                    block_id=block_id,
                    line_id=block.get("block_order"),  # JSON里可能是 null
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
        return len(bbox) == 4

    def _to_bounding_box(self, bbox: Sequence[Any]) -> BoundingBox:
        x1, y1, x2, y2 = bbox
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
