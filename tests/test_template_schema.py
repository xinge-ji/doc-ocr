from __future__ import annotations

from datetime import date

from app.services.rules.template_schema import validate_template_payload


def test_template_schema_validates_and_coerces_values() -> None:
    fields = [
        {
            "name": "invoice_number",
            "path": "invoice_number",
            "type": "string",
            "required": True,
            "constraints": {"regex": "^INV"},
        },
        {
            "name": "amount",
            "path": "amount",
            "type": "number",
            "required": True,
            "constraints": {"min": 0},
        },
        {
            "name": "issue_date",
            "path": "issue_date",
            "type": "date",
            "required": False,
        },
        {
            "name": "lines",
            "path": "lines",
            "type": "array",
            "required": True,
            "items": {
                "type": "object",
                "properties": [
                    {"name": "name", "path": "name", "type": "string", "required": True},
                    {"name": "amount", "path": "amount", "type": "number", "required": True},
                ],
            },
        },
    ]
    payload = {
        "invoice_number": "INV-001",
        "amount": "$1,234.50",
        "issue_date": "2024-01-02",
        "lines": [{"name": "Item", "amount": "100"}],
    }

    result = validate_template_payload(payload, fields)

    assert result.errors == []
    assert result.data["invoice_number"] == "INV-001"
    assert result.data["amount"] == 1234.5
    assert result.data["issue_date"] == date(2024, 1, 2)
    assert result.data["lines"][0]["amount"] == 100.0


def test_template_schema_reports_missing_required_fields() -> None:
    fields = [
        {"name": "invoice_number", "path": "invoice_number", "type": "string", "required": True},
        {"name": "amount", "path": "amount", "type": "number", "required": True},
    ]
    payload = {"invoice_number": ""}

    result = validate_template_payload(payload, fields)

    assert result.errors


def test_template_schema_ignores_invalid_optional_fields_in_array_items() -> None:
    fields = [
        {
            "name": "lines",
            "path": "lines",
            "type": "array",
            "required": True,
            "items": {
                "type": "object",
                "properties": [
                    {"name": "name", "path": "name", "type": "string", "required": True},
                    {"name": "amount", "path": "amount", "type": "number", "required": True},
                    {"name": "quantity", "path": "quantity", "type": "number", "required": False},
                ],
            },
        }
    ]
    payload = {"lines": [{"name": "Item", "amount": "100", "quantity": "1 2"}]}

    result = validate_template_payload(payload, fields)

    assert result.errors == []
    assert result.data["lines"][0]["amount"] == 100.0
