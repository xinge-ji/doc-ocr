"""Usage: validate extracted invoice data against template field definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationResult:
    data: dict[str, Any]
    errors: list[str]


def validate_template_payload(
    payload: dict[str, Any],
    fields: list[Mapping[str, Any]],
) -> ValidationResult:
    normalized: dict[str, Any] = {}
    errors: list[str] = []

    for field in fields:
        path = field.get("path") or field.get("name")
        if not path:
            continue
        raw_value = _get_path(payload, path)
        if _is_empty(raw_value):
            if field.get("required", False):
                errors.append(f"missing_required:{path}")
            continue

        value, field_errors = _coerce_field_value(raw_value, field, path)
        if field_errors:
            errors.extend(field_errors)
            continue
        _set_path(normalized, path, value)

    return ValidationResult(data=normalized, errors=errors)


def _coerce_field_value(
    value: Any,
    field: Mapping[str, Any],
    path: str,
) -> tuple[Any, list[str]]:
    field_type = (field.get("type") or "string").lower()
    errors: list[str] = []

    allowed_types = {"string", "number", "integer", "boolean", "date", "object", "array"}
    if field_type not in allowed_types:
        return None, [f"unknown_type:{path}"]

    if field_type == "object":
        return _coerce_object(value, field, path)
    if field_type == "array":
        return _coerce_array(value, field, path)

    if field_type == "string":
        coerced = str(value).strip()
    elif field_type == "number":
        coerced = _parse_number(value)
    elif field_type == "integer":
        coerced = _parse_integer(value)
    elif field_type == "boolean":
        coerced = _parse_boolean(value)
    elif field_type == "date":
        coerced = _parse_date(value)
    else:
        coerced = value

    if coerced is None:
        errors.append(f"invalid_type:{path}")
        return None, errors

    errors.extend(_apply_constraints(coerced, field.get("constraints") or {}, path))
    return coerced, errors


def _coerce_object(
    value: Any,
    field: Mapping[str, Any],
    path: str,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict):
        return {}, [f"invalid_type:{path}"]

    properties = field.get("properties") or []
    normalized: dict[str, Any] = {}
    errors: list[str] = []
    for prop in properties:
        prop_path = prop.get("path") or prop.get("name")
        if not prop_path:
            continue
        prop_value = value.get(prop_path)
        if _is_empty(prop_value):
            if prop.get("required", False):
                errors.append(f"missing_required:{path}.{prop_path}")
            continue
        coerced, prop_errors = _coerce_field_value(prop_value, prop, f"{path}.{prop_path}")
        if prop_errors:
            errors.extend(prop_errors)
            continue
        normalized[prop_path] = coerced
    return normalized, errors


def _coerce_array(value: Any, field: Mapping[str, Any], path: str) -> tuple[list[Any], list[str]]:
    if not isinstance(value, list):
        return [], [f"invalid_type:{path}"]
    items_def = field.get("items") or {}
    normalized: list[Any] = []
    errors: list[str] = []

    for idx, item in enumerate(value):
        item_path = f"{path}[{idx}]"
        coerced, item_errors = _coerce_field_value(item, items_def, item_path)
        if item_errors:
            errors.extend(item_errors)
            continue
        normalized.append(coerced)

    if field.get("required", False) and not normalized:
        errors.append(f"missing_required:{path}")

    errors.extend(_apply_constraints(normalized, field.get("constraints") or {}, path))
    return normalized, errors


def _apply_constraints(value: Any, constraints: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if not constraints:
        return errors

    regex = constraints.get("regex")
    if regex:
        if not re.search(regex, str(value)):
            errors.append(f"regex_failed:{path}")

    enum = constraints.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"enum_failed:{path}")

    min_value = constraints.get("min")
    max_value = constraints.get("max")
    if isinstance(value, (int, float)):
        if min_value is not None and value < float(min_value):
            errors.append(f"min_failed:{path}")
        if max_value is not None and value > float(max_value):
            errors.append(f"max_failed:{path}")

    min_len = constraints.get("min_len")
    max_len = constraints.get("max_len")
    if isinstance(value, (str, list)):
        length = len(value)
        if min_len is not None and length < int(min_len):
            errors.append(f"min_len_failed:{path}")
        if max_len is not None and length > int(max_len):
            errors.append(f"max_len_failed:{path}")

    return errors


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    cleaned = re.sub(r"[^\d\.-]", "", value_str)
    if cleaned in {"", ".", "-", "-.", ".-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_integer(value: Any) -> int | None:
    number = _parse_number(value)
    if number is None:
        return None
    return int(number)


def _parse_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y"}:
        return True
    if value_str in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    value_str = str(value).strip()
    if not value_str:
        return None
    patterns = [
        r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})",
        r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日?",
    ]
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            year = int(match.group("y"))
            month = int(match.group("m"))
            day = int(match.group("d"))
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def _get_path(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False
