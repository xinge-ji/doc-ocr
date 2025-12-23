from __future__ import annotations

import pytest

from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.pipelines.invoice import InvoiceExtractionPipeline
from app.services.rules.invoice_rule_extractor import InvoiceRuleExtractor


class FakeOcrClient:
    def __init__(self, result: OcrResult) -> None:
        self._result = result

    async def extract(self, *_args, **_kwargs) -> OcrResult:
        return self._result


def _item(text: str, x1: float, y1: float, x2: float, y2: float, page: int = 1) -> OcrItem:
    return OcrItem(
        text=text,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page=page,
    )


def _build_items(include_sum_row: bool = True) -> list[OcrItem]:
    items = [
        _item("电子发票", 100, 10, 160, 25),
        _item("增值税", 180, 10, 220, 25),
        _item("专用发票", 230, 10, 280, 25),
        _item("发票号码", 350, 10, 410, 25),
        _item("12345678", 420, 10, 470, 25),
        _item("销售方名称", 50, 80, 120, 95),
        _item("某某公司", 130, 80, 200, 95),
        _item("项目名称", 50, 200, 110, 215),
        _item("规格型号", 120, 200, 180, 215),
        _item("单位", 190, 200, 220, 215),
        _item("数量", 230, 200, 260, 215),
        _item("单价", 270, 200, 310, 215),
        _item("金额", 320, 200, 360, 215),
        _item("税率/征收率", 370, 200, 440, 215),
        _item("税额", 450, 200, 490, 215),
        _item("服务费", 50, 230, 110, 245),
        _item("100.00", 320, 230, 360, 245),
        _item("13.00", 450, 230, 480, 245),
        _item("开票人", 400, 900, 440, 915),
        _item("张三", 450, 900, 480, 915),
    ]
    if include_sum_row:
        items.extend(
            [
                _item("合", 50, 260, 60, 275),
                _item("计", 70, 260, 80, 275),
                _item("100.00", 320, 260, 360, 275),
                _item("13.00", 450, 260, 480, 275),
            ]
        )
    return items


@pytest.mark.asyncio
async def test_pipeline_skips_llm_when_rule_complete() -> None:
    ocr_result = OcrResult(items=_build_items(include_sum_row=True))
    ocr_client = FakeOcrClient(ocr_result)
    pipeline = InvoiceExtractionPipeline(
        ocr_client=ocr_client,
        rule_extractor=InvoiceRuleExtractor(),
    )

    result = await pipeline.run(b"dummy")

    assert result.complete is True
    assert result.template_name == "vat_special_einvoice"


@pytest.mark.asyncio
async def test_pipeline_reports_failure_when_rule_incomplete() -> None:
    ocr_result = OcrResult(items=_build_items(include_sum_row=False))
    ocr_client = FakeOcrClient(ocr_result)
    pipeline = InvoiceExtractionPipeline(
        ocr_client=ocr_client,
        rule_extractor=InvoiceRuleExtractor(),
    )

    result = await pipeline.run(b"dummy")

    assert result.complete is False
    assert "table_extraction_failed" in result.errors
