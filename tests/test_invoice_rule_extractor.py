from __future__ import annotations

from datetime import date

from app.schemas.ocr import BoundingBox, OcrItem, OcrResult
from app.services.rules.invoice_rule_extractor import InvoiceRuleExtractor


def _item(text: str, x1: float, y1: float, x2: float, y2: float, page: int = 1) -> OcrItem:
    return OcrItem(
        text=text,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page=page,
    )


def _build_items(include_sum_row: bool = True) -> list[OcrItem]:
    items = [
        _item("电子发票", 100, 10, 160, 25),
        _item("（", 170, 10, 175, 25),
        _item("增值税", 180, 10, 220, 25),
        _item("专用发票", 230, 10, 280, 25),
        _item("）", 285, 10, 290, 25),
        _item("发票号码", 350, 10, 410, 25),
        _item("12345678", 420, 10, 470, 25),
        _item("开票日期", 500, 10, 560, 25),
        _item("2024-01-02", 570, 10, 650, 25),
        _item("销售方名称", 50, 80, 120, 95),
        _item("某某公司", 130, 80, 200, 95),
        _item("项目名称", 50, 200, 110, 215),
        _item("规格型号", 120, 200, 180, 215),
        _item("单位", 190, 200, 220, 215),
        _item("数量", 230, 200, 260, 215),
        _item("单价", 270, 200, 310, 215),
        _item("金额", 320, 200, 360, 215),
        _item("税率/征收", 370, 200, 430, 215),
        _item("率", 432, 200, 440, 215),
        _item("税额", 450, 200, 490, 215),
        _item("服务费", 50, 230, 110, 245),
        _item("项", 190, 230, 220, 245),
        _item("1", 230, 230, 240, 245),
        _item("100.00", 270, 230, 310, 245),
        _item("100.00", 320, 230, 360, 245),
        _item("13%", 370, 230, 400, 245),
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


def test_rule_extractor_handles_split_title_and_sum_row() -> None:
    ocr_result = OcrResult(items=_build_items(include_sum_row=True))
    extractor = InvoiceRuleExtractor()

    result = extractor.extract(ocr_result)

    assert result.complete is True
    assert result.data is not None
    assert result.template_name == "vat_special_einvoice"
    assert result.data["invoice_number"] == "12345678"
    assert result.data["issue_date"] == date(2024, 1, 2)
    assert result.data["amount_with_tax"] == 113.0
    assert result.data["lines"]
    assert result.data["lines"][0]["name"] == "服务费"


def test_rule_extractor_merges_anchor_rows() -> None:
    items = [
        _item("电子发票", 100, 10, 160, 25),
        _item("增值税", 180, 10, 220, 25),
        _item("专用发票", 230, 10, 280, 25),
        _item("发票号码", 350, 10, 410, 25),
        _item("12345678", 420, 10, 470, 25),
        _item("开票日期", 500, 10, 560, 25),
        _item("2024-01-02", 570, 10, 650, 25),
        _item("项目名称", 50, 200, 110, 215),
        _item("规格型号", 120, 200, 180, 215),
        _item("单位", 190, 200, 220, 215),
        _item("数量", 230, 200, 260, 215),
        _item("单价", 270, 200, 310, 215),
        _item("金额", 320, 200, 360, 215),
        _item("税率/征收率", 370, 200, 440, 215),
        _item("税额", 450, 200, 490, 215),
        _item("*企业管理服务*公摊电费", 50, 230, 170, 245),
        _item("24-08-01至25-1", 120, 230, 200, 245),
        _item("96.32", 320, 250, 360, 265),
        _item("6%", 370, 250, 400, 265),
        _item("5.78", 450, 250, 480, 265),
        _item("*水冰雪*公摊水费", 50, 280, 150, 295),
        _item("24-07-26至25-1", 120, 280, 220, 295),
        _item("18.55", 320, 300, 360, 315),
        _item("3%", 370, 300, 400, 315),
        _item("0.56", 450, 300, 480, 315),
        _item("合计", 50, 330, 90, 345),
        _item("100.00", 320, 330, 360, 345),
        _item("6.34", 450, 330, 480, 345),
        _item("开票人", 400, 900, 440, 915),
        _item("张三", 450, 900, 480, 915),
    ]
    ocr_result = OcrResult(items=items)
    extractor = InvoiceRuleExtractor()

    result = extractor.extract(ocr_result)

    assert result.complete is True
    assert result.data is not None
    assert len(result.data["lines"]) == 2
    assert result.data["lines"][0]["name"] == "*企业管理服务*公摊电费"
    assert result.data["lines"][0]["amount"] == 96.32


def test_rule_extractor_splits_buyer_seller_by_region() -> None:
    items = _build_items(include_sum_row=True)
    items.extend(
        [
            _item("购买方信息", 10, 110, 30, 180),
            _item("厦门鹭燕医疗器械有限公司", 120, 120, 320, 135),
            _item("91350200705497029M", 120, 150, 260, 165),
            _item("销售方信息", 450, 110, 470, 180),
            _item("福州市为齿而来科技有限公司", 600, 120, 900, 135),
            _item("91350104MAD896RR6Y", 600, 150, 760, 165),
        ]
    )
    ocr_result = OcrResult(items=items)
    extractor = InvoiceRuleExtractor()

    result = extractor.extract(ocr_result)

    assert result.complete is True
    assert result.data is not None
    assert result.data["buyer"]["name"] == "厦门鹭燕医疗器械有限公司"
    assert result.data["buyer"]["tax_id"] == "91350200705497029M"
    assert result.data["seller"]["name"] == "福州市为齿而来科技有限公司"
    assert result.data["seller"]["tax_id"] == "91350104MAD896RR6Y"


def test_rule_extractor_requires_sum_row() -> None:
    ocr_result = OcrResult(items=_build_items(include_sum_row=False))
    extractor = InvoiceRuleExtractor()

    result = extractor.extract(ocr_result)

    assert result.complete is False
    assert result.data is None
