"""Usage: rule-based invoice extraction from OCR items."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from app.schemas.ocr import OcrItem, OcrResult
from app.services.rules.template_loader import load_templates
from app.services.rules.template_matcher import match_template
from app.services.rules.template_schema import validate_template_payload
from app.services.rules.text_normalize import (
    Line,
    MergeTokensConfig,
    NormalizeConfig,
    cluster_lines,
    join_tokens,
    line_text,
    merge_tokens,
    normalize_text,
    to_tokens,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RuleExtractionResult:
    complete: bool
    data: dict[str, Any] | None
    template_name: str | None
    errors: list[str]


class InvoiceRuleExtractor:
    def __init__(self) -> None:
        self._templates = load_templates()

    def extract(self, ocr_result: OcrResult) -> RuleExtractionResult:
        if not ocr_result.items:
            logger.warning("Rule extraction failed: no_ocr_items")
            return RuleExtractionResult(
                complete=False,
                data=None,
                template_name=None,
                errors=["no_ocr_items"],
            )
        if not self._templates:
            logger.warning("Rule extraction failed: no_templates")
            return RuleExtractionResult(
                complete=False,
                data=None,
                template_name=None,
                errors=["no_templates"],
            )

        match = match_template(ocr_result.items, self._templates)
        if not match:
            logger.warning("Rule extraction failed: template_not_matched")
            return RuleExtractionResult(
                complete=False,
                data=None,
                template_name=None,
                errors=["template_not_matched"],
            )

        template = match.template
        page_items = [item for item in ocr_result.items if item.page == match.page]
        page_width, page_height = _page_bounds(page_items)

        title_rule = template.get("match_rules", {}).get("title") or {}
        lines = cluster_lines(page_items, y_tol=float(title_rule.get("y_tol", 6)))
        title_line = match.title_line or _best_title_line(lines)

        payload: dict[str, Any] = {}
        payload.update(template.get("fixed_fields", {}))

        non_table_fields = template.get("non_table_fields") or []
        for field in non_table_fields:
            value = self._extract_field(field, lines, page_width, page_height, title_line)
            if value is not None:
                _set_path(payload, field.get("target_path") or field.get("name"), value)

        table_cfg = template.get("table") or {}
        table_result = self._extract_table(table_cfg, lines)
        template_name = template.get("name")
        if table_result is None:
            logger.warning("Rule extraction failed: table_extraction_failed template=%s", template_name)
            return RuleExtractionResult(
                complete=False,
                data=None,
                template_name=template_name,
                errors=["table_extraction_failed"],
            )

        payload["lines"] = table_result.lines
        sum_targets = table_cfg.get("sum_row", {}).get("targets") or {}
        for key, target in sum_targets.items():
            if key in table_result.sum_values:
                _set_path(payload, target, table_result.sum_values[key])

        validation = validate_template_payload(payload, template.get("fields") or [])
        if validation.errors:
            logger.warning(
                "Rule extraction failed: validation_failed template=%s errors=%s",
                template_name,
                validation.errors,
            )
            return RuleExtractionResult(
                complete=False,
                data=None,
                template_name=template_name,
                errors=validation.errors,
            )

        return RuleExtractionResult(
            complete=True,
            data=validation.data,
            template_name=template_name,
            errors=[],
        )

    def _extract_field(
        self,
        field: Mapping[str, Any],
        lines: list[Line],
        page_width: float,
        page_height: float,
        title_line: Line | None,
    ) -> str | None:
        use = (field.get("use") or "text").lower()
        use_text = use in {"text", "both"}
        use_pos = use in {"pos", "both"}
        normalize_cfg = NormalizeConfig.from_dict(
            field.get("normalize"),
            default_remove_whitespace=True,
        )
        field_name = field.get("name") or field.get("target_path") or "unknown"

        if use_text:
            value = self._extract_by_text(
                field,
                lines,
                page_width,
                page_height,
                title_line,
                normalize_cfg,
            )
            if value:
                logger.debug("Rule field extracted: %s method=text", field_name)
                return value
        if use_pos:
            value = self._extract_by_pos(
                field,
                page_width,
                page_height,
                title_line,
                normalize_cfg,
                lines,
            )
            if value:
                logger.debug("Rule field extracted: %s method=pos", field_name)
                return value
        logger.debug("Rule field missing: %s use=%s", field_name, use)
        return None

    def _extract_by_text(
        self,
        field: Mapping[str, Any],
        lines: list[Line],
        page_width: float,
        page_height: float,
        title_line: Line | None,
        normalize_cfg: NormalizeConfig,
    ) -> str | None:
        anchor = field.get("anchor_text") or {}
        pattern = anchor.get("pattern")
        match_scope = (anchor.get("match_scope") or "line").lower()
        value_from = anchor.get("value_from") or "capture"
        region_lines = lines
        anchor_pos = field.get("anchor_pos") or {}
        region_bounds = _resolve_region_bounds(anchor_pos, page_width, page_height, title_line)
        if region_bounds:
            region_items = _filter_items_in_region(
                [item for line in lines for item in line.items],
                region_bounds,
            )
            if not region_items:
                return None
            y_tol = float(anchor.get("y_tol", 6))
            region_lines = cluster_lines(region_items, y_tol=y_tol)
        if pattern:
            return _extract_with_pattern(
                region_lines,
                pattern,
                match_scope,
                value_from,
                normalize_cfg,
            )

        anchor_text = anchor.get("text")
        if not anchor_text:
            return None
        anchor_norm = normalize_text(str(anchor_text), normalize_cfg)
        y_tol = float(anchor.get("y_tol", 6))
        x_gap = anchor.get("x_gap") or [0, float("inf")]
        x_min, x_max = float(x_gap[0]), float(x_gap[1])

        for line in region_lines:
            items = line.sorted_items()
            for idx, item in enumerate(items):
                item_norm = normalize_text(item.text, normalize_cfg)
                if anchor_norm not in item_norm:
                    continue
                tail = _strip_anchor_tail(item.text, anchor_text)
                if tail:
                    return tail
                anchor_bbox = item.bounding_box
                candidates = []
                for cand in items[idx + 1 :]:
                    gap = cand.bounding_box.x1 - anchor_bbox.x2
                    if gap < x_min or gap > x_max:
                        continue
                    if abs(cand.bounding_box.y1 - anchor_bbox.y1) <= y_tol:
                        candidates.append(cand)
                if candidates:
                    candidates.sort(key=lambda c: c.bounding_box.x1)
                    return candidates[0].text.strip()
                if anchor.get("fallback_right_neighbor"):
                    return _nearest_right_neighbor(items, anchor_bbox, y_tol)
        return None

    def _extract_by_pos(
        self,
        field: Mapping[str, Any],
        page_width: float,
        page_height: float,
        title_line: Line | None,
        normalize_cfg: NormalizeConfig,
        lines: list[Line],
    ) -> str | None:
        anchor_pos = field.get("anchor_pos") or {}
        region_bounds = _resolve_region_bounds(anchor_pos, page_width, page_height, title_line)
        if not region_bounds:
            return None
        items = [item for line in lines for item in line.items]
        region_items = _filter_items_in_region(items, region_bounds)
        if not region_items:
            return None

        tokens = to_tokens(region_items)
        tokens.sort(key=lambda t: t.x1)
        merge_cfg = MergeTokensConfig.from_dict(field.get("merge_tokens"))
        value_regex = field.get("value_regex")
        match_scope = (field.get("pos_match_scope") or anchor_pos.get("match_scope") or "join").lower()
        if match_scope == "line":
            line_y_tol = float(field.get("pos_line_y_tol") or anchor_pos.get("line_y_tol") or 6)
            region_lines = cluster_lines(region_items, y_tol=line_y_tol)
            for line in sorted(region_lines, key=lambda entry: entry.y_center):
                line_tokens = to_tokens(line.sorted_items())
                if merge_cfg.merge_single_char or merge_cfg.max_x_gap > 0:
                    line_tokens = merge_tokens(line_tokens, merge_cfg)
                raw_text = join_tokens(line_tokens)
                if value_regex:
                    match = re.search(value_regex, normalize_text(raw_text, normalize_cfg))
                    if not match:
                        continue
                    value = match.group(1) if match.lastindex else match.group(0)
                    return value.strip()
                if field.get("allow_extra"):
                    return normalize_text(raw_text, normalize_cfg)
                return raw_text.strip()
            return None

        if merge_cfg.merge_single_char or merge_cfg.max_x_gap > 0:
            tokens = merge_tokens(tokens, merge_cfg)
        raw_text = join_tokens(tokens)
        if value_regex:
            match = re.search(value_regex, normalize_text(raw_text, normalize_cfg))
            if not match:
                return None
            value = match.group(1) if match.lastindex else match.group(0)
            return value.strip()
        if field.get("allow_extra"):
            return normalize_text(raw_text, normalize_cfg)
        return raw_text.strip()

    def _extract_table(
        self,
        table_cfg: Mapping[str, Any],
        lines: list[Line],
    ) -> "_TableResult" | None:
        if not table_cfg:
            return None
        header_labels = table_cfg.get("header") or []
        header_match = table_cfg.get("header_match") or {}
        header_norm = NormalizeConfig.from_dict(
            header_match.get("normalize"),
            default_remove_whitespace=True,
        )
        header_merge = MergeTokensConfig.from_dict(header_match.get("merge_tokens"))
        min_hit = int(header_match.get("min_hit", len(header_labels)))

        header_line = None
        header_columns = None
        best_hits: list[str] = []
        best_line: Line | None = None
        best_line_value = ""
        for line in lines:
            line_value = line_text(line, normalize=header_norm)
            hits = [
                label
                for label in header_labels
                if normalize_text(label, header_norm) in line_value
            ]
            if len(hits) > len(best_hits):
                best_hits = hits
                best_line = line
                best_line_value = line_value
            match = _match_header_line(line, header_labels, header_norm, header_merge, min_hit)
            if match:
                header_line = line
                header_columns = match
                break

        if header_line is None or header_columns is None:
            logger.warning("Rule table extraction failed: header_not_found")
            if best_line is not None:
                logger.debug(
                    "Rule table header best_line y=%.2f hits=%s text=%s",
                    best_line.y_center,
                    best_hits,
                    best_line_value,
                )
            return None

        boundary_by_header_end = set(table_cfg.get("boundary_by_header_end") or [])
        first_column_left = table_cfg.get("first_column_left")
        columns = _build_columns(
            header_columns,
            table_cfg.get("column_map") or {},
            boundary_by_header_end,
            first_column_left,
        )
        logger.debug(
            "Rule table header matched labels=%s",
            list(header_columns.keys()),
        )
        assign_rule = table_cfg.get("assign_rule") or {}
        x_tol = float(assign_rule.get("x_tol", 0))
        if x_tol:
            for column in columns:
                column["left"] -= x_tol
                column["right"] += x_tol

        row_group = table_cfg.get("row_group") or {}
        mode = (row_group.get("mode") or "line").lower()
        row_y_gap = float(row_group.get("y_gap", 8))
        allow_blank = bool(row_group.get("allow_blank", True))
        blank_row_max = int(row_group.get("blank_row_max", 3))
        ignore_blank = bool(row_group.get("ignore_blank", False))
        anchor_required = set(row_group.get("anchor_required") or ["name"])
        anchor_any = set(row_group.get("anchor_any") or [])
        anchor_skip_before_sum = bool(row_group.get("anchor_skip_before_sum", False))
        merge_join_fields = set(row_group.get("merge_join") or ["name", "spec", "unit"])
        merge_first_fields = set(
            row_group.get("merge_first")
            or ["quantity", "unit_price", "amount", "tax_rate", "tax_amount"]
        )
        merge_joiner = str(row_group.get("joiner") or "")
        stop_anchors = table_cfg.get("row_end", {}).get("stop_anchors") or []
        sum_row_cfg = table_cfg.get("sum_row") or {}
        sum_required = bool(sum_row_cfg.get("required", False))
        sum_norm = NormalizeConfig.from_dict(
            sum_row_cfg.get("normalize"),
            default_remove_whitespace=True,
        )
        sum_merge = MergeTokensConfig.from_dict(sum_row_cfg.get("merge_tokens"))
        neighbor_cfg = sum_row_cfg.get("neighbor_search") or {}
        neighbor_up = int(neighbor_cfg.get("max_lines_up", 0))
        neighbor_down = int(neighbor_cfg.get("max_lines_down", 0))
        stop_anchor = neighbor_cfg.get("stop_anchor")
        stop_norm = NormalizeConfig(
            remove_whitespace=True,
            remove_brackets=True,
            fullwidth_to_halfwidth=sum_norm.fullwidth_to_halfwidth,
            lowercase=sum_norm.lowercase,
        )
        required_fields = set(table_cfg.get("required_fields") or [])

        lines_sorted = sorted(lines, key=lambda line: line.y_center)
        start_index = lines_sorted.index(header_line) + 1
        blank_rows = 0
        line_items: list[dict[str, Any]] = []
        sum_values: dict[str, float] = {}
        current_block: list[dict[str, str]] | None = None
        column_fields = [column["field"] for column in columns]

        for idx in range(start_index, len(lines_sorted)):
            line = lines_sorted[idx]
            if line.y_center <= header_line.y_center + row_y_gap:
                continue
            line_value = line_text(line, normalize=sum_norm)
            if _contains_stop_anchor(line_value, stop_anchors, sum_norm):
                break

            row_cells = _assign_row_cells(line, columns)
            if _is_sum_row(line, sum_row_cfg, sum_norm, sum_merge):
                logger.debug(
                    "Rule table sum row detected y=%.2f cells=%s",
                    line.y_center,
                    row_cells,
                )
                amount_val = _parse_number(row_cells.get("amount"))
                tax_val = _parse_number(row_cells.get("tax_amount"))
                if (amount_val is None or tax_val is None) and (neighbor_up > 0 or neighbor_down > 0):
                    amount_val, tax_val = _fill_sum_from_neighbors(
                        lines_sorted,
                        idx,
                        start_index,
                        columns,
                        sum_norm,
                        stop_anchor,
                        stop_norm,
                        neighbor_up,
                        neighbor_down,
                        amount_val,
                        tax_val,
                    )
                if amount_val is None and sum_required:
                    logger.warning("Rule table extraction failed: sum_row_missing_amount")
                    return None
                logger.debug("Rule table sum row cells=%s", row_cells)
                if amount_val is not None:
                    sum_values["amount"] = amount_val
                if tax_val is not None:
                    sum_values["tax_amount"] = tax_val
                if amount_val is not None and tax_val is not None:
                    sum_values["amount_with_tax"] = amount_val + tax_val
                elif amount_val is not None:
                    sum_values["amount_with_tax"] = amount_val
                break

            if mode == "anchor":
                if not any(row_cells.values()):
                    if ignore_blank:
                        continue
                    if not allow_blank:
                        break
                    blank_rows += 1
                    if blank_rows >= blank_row_max:
                        break
                    continue

                blank_rows = 0
                is_anchor = _is_anchor_row(row_cells, anchor_required, anchor_any)
                if (
                    is_anchor
                    and anchor_skip_before_sum
                    and current_block is not None
                    and _next_effective_line_is_sum(
                        lines_sorted,
                        idx,
                        start_index,
                        columns,
                        sum_row_cfg,
                        sum_norm,
                        sum_merge,
                        stop_anchors,
                        ignore_blank,
                    )
                ):
                    current_block.append(row_cells)
                    logger.debug(
                        "Rule table anchor skipped before sum y=%.2f cells=%s",
                        line.y_center,
                        row_cells,
                    )
                    continue
                if is_anchor:
                    if current_block:
                        merged = _merge_row_cells(
                            current_block,
                            column_fields,
                            merge_join_fields,
                            merge_first_fields,
                            merge_joiner,
                        )
                        _append_row_payload(line_items, merged, required_fields)
                    current_block = [row_cells]
                    logger.debug(
                        "Rule table anchor row started y=%.2f cells=%s",
                        line.y_center,
                        row_cells,
                    )
                else:
                    if current_block is None:
                        logger.debug("Rule table row skipped before anchor cells=%s", row_cells)
                        continue
                    current_block.append(row_cells)
                continue

            if not any(row_cells.values()):
                if not allow_blank:
                    break
                blank_rows += 1
                if blank_rows >= blank_row_max:
                    break
                continue

            blank_rows = 0
            _append_row_payload(line_items, row_cells, required_fields)

        if mode == "anchor" and current_block:
            merged = _merge_row_cells(
                current_block,
                column_fields,
                merge_join_fields,
                merge_first_fields,
                merge_joiner,
            )
            _append_row_payload(line_items, merged, required_fields)

        if sum_required and "amount_with_tax" not in sum_values:
            logger.warning("Rule table extraction failed: sum_row_missing")
            return None
        if not line_items:
            logger.warning("Rule table extraction warning: no_lines")
            return _TableResult(lines=[], sum_values=sum_values)

        logger.debug(
            "Rule table extracted rows=%d sum_found=%s",
            len(line_items),
            "amount_with_tax" in sum_values,
        )
        return _TableResult(lines=line_items, sum_values=sum_values)


@dataclass(frozen=True)
class _TableResult:
    lines: list[dict[str, Any]]
    sum_values: dict[str, float]


def _extract_with_pattern(
    lines: list[Line],
    pattern: str,
    match_scope: str,
    value_from: str,
    normalize_cfg: NormalizeConfig,
) -> str | None:
    for line in lines:
        items = line.sorted_items()
        if match_scope == "box":
            for item in items:
                match = re.search(pattern, normalize_text(item.text, normalize_cfg))
                if match:
                    return _value_from_match(match, value_from)
        else:
            line_value = normalize_text(join_tokens(to_tokens(items)), normalize_cfg)
            match = re.search(pattern, line_value)
            if match:
                return _value_from_match(match, value_from)
    return None


def _value_from_match(match: re.Match[str], value_from: str) -> str:
    if value_from == "capture" and match.lastindex:
        return match.group(match.lastindex).strip()
    return match.group(0).strip()


def _strip_anchor_tail(raw: str, anchor_text: str) -> str | None:
    if anchor_text not in raw:
        return None
    tail = raw.split(anchor_text, 1)[-1]
    tail = re.sub(r"^[\u003a\uFF1A]+", "", tail).strip()
    return tail or None


def _nearest_right_neighbor(items: list[OcrItem], anchor_bbox: Any, y_tol: float) -> str | None:
    candidates = []
    for item in items:
        if item.bounding_box.x1 <= anchor_bbox.x2:
            continue
        if abs(item.bounding_box.y1 - anchor_bbox.y1) > y_tol:
            continue
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.bounding_box.x1)
    return candidates[0].text.strip() or None


def _best_title_line(lines: list[Line]) -> Line | None:
    if not lines:
        return None
    return min(lines, key=lambda line: line.y_center)


def _match_header_line(
    line: Line,
    header_labels: list[str],
    normalize_cfg: NormalizeConfig,
    merge_cfg: MergeTokensConfig,
    min_hit: int,
) -> dict[str, tuple[float, float, float, float]] | None:
    tokens = to_tokens(line.sorted_items())
    if merge_cfg.merge_single_char or merge_cfg.max_x_gap > 0:
        tokens = merge_tokens(tokens, merge_cfg)
    token_values = [normalize_text(token.text, normalize_cfg) for token in tokens]
    matches: dict[str, tuple[float, float, float, float]] = {}
    start_idx = 0
    for label in header_labels:
        label_norm = normalize_text(label, normalize_cfg)
        found = None
        for i in range(start_idx, len(tokens)):
            acc = ""
            for j in range(i, len(tokens)):
                acc += token_values[j]
                if acc == label_norm or label_norm in acc:
                    found = (i, j)
                    break
            if found:
                break
        if not found:
            break
        i, j = found
        span = tokens[i : j + 1]
        x1 = min(token.x1 for token in span)
        y1 = min(token.y1 for token in span)
        x2 = max(token.x2 for token in span)
        y2 = max(token.y2 for token in span)
        matches[label] = (x1, y1, x2, y2)
        start_idx = j + 1

    if len(matches) < min_hit:
        return None
    return matches


def _build_columns(
    header_columns: dict[str, tuple[float, float, float, float]],
    column_map: Mapping[str, str],
    boundary_by_header_end: set[str] | None = None,
    first_column_left: float | None = None,
) -> list[dict[str, Any]]:
    entries = []
    boundary_labels = boundary_by_header_end or set()
    for label, (x1, _y1, x2, _y2) in header_columns.items():
        center = (x1 + x2) / 2
        entries.append((center, label, column_map.get(label, label), x1, x2))
    entries.sort(key=lambda e: e[0])

    columns: list[dict[str, Any]] = []
    for idx, (center, label, field, x1, x2) in enumerate(entries):
        if idx > 0 and label in boundary_labels:
            left = entries[idx - 1][4]
        else:
            left = (entries[idx - 1][0] + center) / 2 if idx > 0 else x1
        if label in boundary_labels:
            right = x2
        else:
            right = (center + entries[idx + 1][0]) / 2 if idx < len(entries) - 1 else x2
        if idx == 0 and first_column_left is not None:
            left = float(first_column_left)
        columns.append({"label": label, "field": field, "left": left, "right": right})
    return columns


def _assign_row_cells(line: Line, columns: list[dict[str, Any]]) -> dict[str, str]:
    cells: dict[str, str] = {column["field"]: "" for column in columns}
    tokens = to_tokens(line.sorted_items())
    for token in tokens:
        for column in columns:
            if column["left"] <= token.x_center <= column["right"]:
                cells[column["field"]] += token.text
                break
    return {key: value.strip() for key, value in cells.items()}


def _is_anchor_row(
    row_cells: Mapping[str, str],
    required_fields: set[str],
    any_fields: set[str],
) -> bool:
    if required_fields and not all(row_cells.get(field) for field in required_fields):
        return False
    if not any_fields:
        return True
    return any(row_cells.get(field) for field in any_fields)


def _merge_row_cells(
    rows: list[Mapping[str, str]],
    column_fields: list[str],
    merge_join_fields: set[str],
    merge_first_fields: set[str],
    joiner: str,
) -> dict[str, str]:
    merged: dict[str, str] = {field: "" for field in column_fields}
    for field in column_fields:
        values = [row.get(field, "").strip() for row in rows if row.get(field)]
        if not values:
            continue
        if field in merge_join_fields:
            merged[field] = joiner.join(values)
        elif field in merge_first_fields:
            merged[field] = values[0]
        else:
            merged[field] = values[0]
    return merged


def _append_row_payload(
    line_items: list[dict[str, Any]],
    row_cells: Mapping[str, str],
    required_fields: set[str],
) -> None:
    line_payload = {key: value for key, value in row_cells.items() if value}
    if required_fields and not required_fields.issubset(
        {key for key, value in line_payload.items() if value}
    ):
        logger.warning(
            "Rule table row dropped missing required fields=%s cells=%s",
            sorted(required_fields),
            row_cells,
        )
        return
    line_items.append(line_payload)


def _next_effective_line_is_sum(
    lines: list[Line],
    idx: int,
    start_index: int,
    columns: list[dict[str, Any]],
    sum_cfg: Mapping[str, Any],
    sum_norm: NormalizeConfig,
    sum_merge: MergeTokensConfig,
    stop_anchors: list[str],
    ignore_blank: bool,
) -> bool:
    for next_idx in range(idx + 1, len(lines)):
        line = lines[next_idx]
        line_value = line_text(line, normalize=sum_norm)
        if _contains_stop_anchor(line_value, stop_anchors, sum_norm):
            return False
        if _is_sum_row(line, sum_cfg, sum_norm, sum_merge):
            return True
        row_cells = _assign_row_cells(line, columns)
        if any(row_cells.values()):
            return False
        if not ignore_blank:
            return False
        if next_idx >= start_index:
            continue
    return False


def _fill_sum_from_neighbors(
    lines: list[Line],
    idx: int,
    start_index: int,
    columns: list[dict[str, Any]],
    sum_norm: NormalizeConfig,
    stop_anchor: str | None,
    stop_norm: NormalizeConfig,
    max_up: int,
    max_down: int,
    amount_val: float | None,
    tax_val: float | None,
) -> tuple[float | None, float | None]:
    if max_down > 0:
        for offset in range(1, max_down + 1):
            next_idx = idx + offset
            if next_idx >= len(lines):
                break
            line = lines[next_idx]
            if stop_anchor:
                line_value = line_text(line, normalize=stop_norm)
                if _contains_stop_anchor(line_value, [stop_anchor], stop_norm):
                    logger.debug(
                        "Rule table sum neighbor stop at y=%.2f anchor=%s",
                        line.y_center,
                        stop_anchor,
                    )
                    break
            row_cells = _assign_row_cells(line, columns)
            logger.debug("Rule table sum neighbor down y=%.2f cells=%s", line.y_center, row_cells)
            if amount_val is None:
                amount_val = _parse_number(row_cells.get("amount"))
            if tax_val is None:
                tax_val = _parse_number(row_cells.get("tax_amount"))
            if amount_val is not None and tax_val is not None:
                return amount_val, tax_val

    if max_up > 0:
        for offset in range(1, max_up + 1):
            prev_idx = idx - offset
            if prev_idx < start_index:
                break
            line = lines[prev_idx]
            row_cells = _assign_row_cells(line, columns)
            logger.debug("Rule table sum neighbor up y=%.2f cells=%s", line.y_center, row_cells)
            if amount_val is None:
                amount_val = _parse_number(row_cells.get("amount"))
            if tax_val is None:
                tax_val = _parse_number(row_cells.get("tax_amount"))
            if amount_val is not None and tax_val is not None:
                return amount_val, tax_val

    return amount_val, tax_val


def _is_sum_row(
    line: Line,
    sum_cfg: Mapping[str, Any],
    normalize_cfg: NormalizeConfig,
    merge_cfg: MergeTokensConfig,
) -> bool:
    key = sum_cfg.get("key")
    if not key:
        return False
    key_norm = normalize_text(key, normalize_cfg)
    tokens = to_tokens(line.sorted_items())
    if merge_cfg.merge_single_char or merge_cfg.max_x_gap > 0:
        tokens = merge_tokens(tokens, merge_cfg)
    for token in tokens:
        if normalize_text(token.text, normalize_cfg) == key_norm:
            return True
    line_value = normalize_text(join_tokens(tokens), normalize_cfg)
    return key_norm in line_value


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = re.sub(r"[^\d\.-]", "", value)
    if not cleaned or cleaned in {".", "-", "-.", ".-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _contains_stop_anchor(value: str, anchors: list[str], normalize_cfg: NormalizeConfig) -> bool:
    for anchor in anchors:
        if normalize_text(anchor, normalize_cfg) in value:
            return True
    return False


def _page_bounds(items: list[OcrItem]) -> tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for item in items:
        max_x = max(max_x, item.bounding_box.x2)
        max_y = max(max_y, item.bounding_box.y2)
    return max_x, max_y


def _resolve_region_bounds(
    anchor_pos: Mapping[str, Any],
    page_width: float,
    page_height: float,
    title_line: Line | None,
) -> tuple[float, float, float, float] | None:
    region = anchor_pos.get("region") or {}
    if not region:
        return None
    relative_to = (anchor_pos.get("relative_to") or "page").lower()
    x_range = region.get("x") or [0.0, 1.0]
    y_range = region.get("y") or [0.0, 1.0]
    x_min, x_max = float(x_range[0]) * page_width, float(x_range[1]) * page_width
    if relative_to == "title_line" and title_line:
        y_center = title_line.y_center
        y_min = y_center + float(y_range[0]) * page_height
        y_max = y_center + float(y_range[1]) * page_height
    else:
        y_min = float(y_range[0]) * page_height
        y_max = float(y_range[1]) * page_height
    return x_min, x_max, y_min, y_max


def _filter_items_in_region(
    items: list[OcrItem],
    bounds: tuple[float, float, float, float],
) -> list[OcrItem]:
    x_min, x_max, y_min, y_max = bounds
    return [
        item
        for item in items
        if x_min <= (item.bounding_box.x1 + item.bounding_box.x2) / 2 <= x_max
        and y_min <= (item.bounding_box.y1 + item.bounding_box.y2) / 2 <= y_max
    ]


def _set_path(payload: dict[str, Any], path: str | None, value: Any) -> None:
    if not path:
        return
    parts = path.split(".")
    cursor = payload
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
