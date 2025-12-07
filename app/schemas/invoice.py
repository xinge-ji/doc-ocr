from datetime import date
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_NUMERIC_CLEAN_RE = re.compile(r"[^\d\.-]")


def _parse_float_like(value: float | str | None, *, allow_none: bool) -> float | None:
    """Normalize currency/number strings like 'CNY 1,000.00' into floats."""

    if value is None:
        if allow_none:
            return None
        raise ValueError("Value is required and cannot be null")

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    if not value_str:
        if allow_none:
            return None
        raise ValueError("Value is required and cannot be empty")

    cleaned = _NUMERIC_CLEAN_RE.sub("", value_str)
    if cleaned in {"", ".", "-"}:
        if allow_none:
            return None
        raise ValueError(f"Cannot parse numeric value from: {value}")

    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric format: {value}") from exc


class BuyerInfo(BaseModel):
    """Purchasing party details captured from an invoice."""

    name: Optional[str] = Field(
        default=None,
        description="购买方名称",
    )
    tax_id: Optional[str] = Field(
        default=None,
        description="购买方统一社会信用代码/纳税人识别号",
    )
    address: Optional[str] = Field(
        default=None,
        description="购买方地址",
    )
    phone: Optional[str] = Field(
        default=None,
        description="购买方电话",
    )
    bank_account_name: Optional[str] = Field(
        default=None,
        description="购买方开户银行",
    )
    bank_account_no: Optional[str] = Field(
        default=None,
        description="购买方银行账号",
    )


class SellerInfo(BaseModel):
    """Selling party details captured from an invoice."""

    name: str = Field(
        ...,
        description="销售方名称",
    )
    tax_id: Optional[str] = Field(
        default=None,
        description="销售方统一社会信用代码/纳税人识别号",
    )
    address: Optional[str] = Field(
        default=None,
        description="销售方地址",
    )
    phone: Optional[str] = Field(
        default=None,
        description="销售方电话",
    )
    bank_account_name: Optional[str] = Field(
        default=None,
        description="销售方开户银行",
    )
    bank_account_no: Optional[str] = Field(
        default=None,
        description="销售方银行账号",
    )


class InvoiceLine(BaseModel):
    """Line item on an invoice containing product/service details."""

    name: str = Field(
        ...,
        description="商品、货物、服务、项目等名称",
    )
    spec: Optional[str] = Field(
        default=None,
        description="规格型号",
    )
    unit: str | None = Field(
        default=None,
        description="单位",
    )
    quantity: float | str | None = Field(
        default=None,
        ge=0,
        description="数量",
    )
    unit_price: float | str | None = Field(
        default=None,
        ge=0,
        description="单价",
    )
    amount: float | str = Field(
        ...,
        ge=0,
        description="金额",
    )
    tax_rate: float | str | None = Field(
        default=None,
        ge=0,
        description="税率 as a decimal (e.g., 0.13 for 13%).",
    )
    tax_amount: float | str | None = Field(
        default=None,
        ge=0,
        description="税额",
    )
    line_total_with_tax: float | str | None = Field(
        default=None,
        ge=0,
        description="Line total when available.",
    )

    @field_validator(
        "quantity",
        "unit_price",
        "tax_rate",
        "tax_amount",
        "line_total_with_tax",
        mode="before",
    )
    @classmethod
    def _parse_optional_numbers(cls, value: float | str | None) -> float | None:
        return _parse_float_like(value, allow_none=True)

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_required_amount(cls, value: float | str | None) -> float:
        parsed = _parse_float_like(value, allow_none=False)
        assert parsed is not None
        return parsed


class InvoiceData(BaseModel):
    """Structured invoice payload extracted via OCR + LLM."""

    invoice_type: str = Field(
        ...,
        description="Invoice category/type (e.g., VAT special, VAT normal).",
    )
    buyer: BuyerInfo = Field(
        default_factory=BuyerInfo,
        description="Purchasing party information block.",
    )
    seller: SellerInfo = Field(
        ...,
        description="Selling party information block.",
    )
    issue_date: date = Field(
        ...,
        description="Invoice issue date.",
    )
    issuer: str = Field(
        ...,
        description="Person who issued the invoice.",
    )
    invoice_number: str = Field(
        ...,
        description="Invoice serial/number field.",
    )
    lines: list[InvoiceLine] = Field(
        default_factory=list,
        description="Collection of item/service line records.",
    )
    total_amount: float | str | None = Field(
        default=None,
        ge=0,
        description="Invoice-level total before tax (总额) when available.",
    )
    amount_with_tax: float | str = Field(
        ...,
        ge=0,
        description="Invoice total including tax (价税合计).",
    )

    @field_validator("total_amount", mode="before")
    @classmethod
    def _parse_total_amount(cls, value: float | str | None) -> float | None:
        return _parse_float_like(value, allow_none=True)

    @field_validator("amount_with_tax", mode="before")
    @classmethod
    def _parse_amount_with_tax(cls, value: float | str | None) -> float:
        parsed = _parse_float_like(value, allow_none=False)
        assert parsed is not None
        return parsed


class ExtractionResponse(BaseModel):
    """Standard envelope for invoice extraction results."""

    success: bool = Field(
        ...,
        description="Indicates whether extraction succeeded end-to-end.",
    )
    data: InvoiceData = Field(
        ...,
        description="Structured invoice fields payload.",
    )
    message: str = Field(
        ...,
        description="Human-readable status or error message.",
    )
