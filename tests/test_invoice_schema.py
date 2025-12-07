from datetime import date

import pytest

from app.schemas.invoice import BuyerInfo, InvoiceData, InvoiceLine, SellerInfo


def test_invoice_line_parses_currency_strings() -> None:
    line = InvoiceLine(
        name="Item",
        amount="¥1,234.56",
        unit_price="1,234.56",
        quantity="2",
        tax_amount="0.00",
    )

    assert line.amount == 1234.56
    assert line.unit_price == 1234.56
    assert line.quantity == 2.0
    assert line.tax_amount == 0.0


def test_invoice_line_rejects_invalid_amount_string() -> None:
    with pytest.raises(ValueError):
        InvoiceLine(
            name="Item",
            amount="not-a-number",
        )


def test_invoice_data_parses_amount_with_tax() -> None:
    payload = InvoiceData(
        invoice_type="vat",
        seller=SellerInfo(name="ACME"),
        issue_date=date(2024, 1, 1),
        issuer="Alice",
        invoice_number="INV-001",
        lines=[],
        amount_with_tax="￥9,999.01",
        total_amount="",
        buyer=BuyerInfo(),
    )

    assert payload.amount_with_tax == 9999.01
    assert payload.total_amount is None
