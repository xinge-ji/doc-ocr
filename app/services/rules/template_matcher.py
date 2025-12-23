"""Usage: match OCR output against invoice templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from app.schemas.ocr import OcrItem
from app.services.rules.text_normalize import (
    Line,
    NormalizeConfig,
    cluster_lines,
    line_text,
    normalize_text,
)


@dataclass(frozen=True)
class TemplateMatch:
    template: dict[str, Any]
    score: int
    page: int
    title_line: Line | None


def match_template(
    ocr_items: Iterable[OcrItem],
    templates: list[dict[str, Any]],
) -> TemplateMatch | None:
    items_by_page: dict[int, list[OcrItem]] = {}
    for item in ocr_items:
        items_by_page.setdefault(item.page, []).append(item)

    best_match: TemplateMatch | None = None
    for template in templates:
        match_rules = template.get("match_rules", {})
        for page, page_items in items_by_page.items():
            candidate = _match_template_on_page(template, match_rules, page_items, page)
            if candidate and (best_match is None or candidate.score > best_match.score):
                best_match = candidate
    return best_match


def _match_template_on_page(
    template: dict[str, Any],
    match_rules: Mapping[str, Any],
    items: list[OcrItem],
    page: int,
) -> TemplateMatch | None:
    title_rule = match_rules.get("title") or {}
    y_tol = float(title_rule.get("y_tol", 6))
    normalize_cfg = NormalizeConfig.from_dict(
        title_rule.get("normalize"),
        default_remove_whitespace=True,
    )

    lines = cluster_lines(items, y_tol=y_tol)
    title_line = _find_title_line(lines, title_rule, normalize_cfg)
    if title_line is None and title_rule.get("required", True):
        return None

    score = int(title_rule.get("score", 0)) if title_line else 0
    anchors = match_rules.get("anchors") or []
    page_width, page_height = _page_bounds(items)
    for anchor in anchors:
        found = _match_anchor(anchor, lines, page_width, page_height)
        if found:
            score += int(anchor.get("score", 0))
        elif anchor.get("required", False):
            return None

    min_score = int(match_rules.get("min_score", 0))
    if score < min_score:
        return None

    return TemplateMatch(template=template, score=score, page=page, title_line=title_line)


def _find_title_line(
    lines: list[Line],
    title_rule: Mapping[str, Any],
    normalize_cfg: NormalizeConfig,
) -> Line | None:
    parts = title_rule.get("parts")
    text = title_rule.get("text")
    order = title_rule.get("order", "left_to_right")
    require_same_line = bool(title_rule.get("same_line", True))

    for line in lines:
        line_value = line_text(line, normalize=normalize_cfg)
        if parts:
            if not require_same_line:
                continue
            if _match_parts_in_order(line_value, parts, normalize_cfg, order):
                return line
        elif text:
            if normalize_text(text, normalize_cfg) in line_value:
                return line
    return None


def _match_parts_in_order(
    line_value: str,
    parts: list[str],
    normalize_cfg: NormalizeConfig,
    order: str,
) -> bool:
    if order != "left_to_right":
        return False
    cursor = 0
    for part in parts:
        part_norm = normalize_text(part, normalize_cfg)
        idx = line_value.find(part_norm, cursor)
        if idx < 0:
            return False
        cursor = idx + len(part_norm)
    return True


def _match_anchor(
    anchor: Mapping[str, Any],
    lines: list[Line],
    page_width: float,
    page_height: float,
) -> bool:
    region = anchor.get("region")
    text_rule = anchor.get("text")
    pattern = anchor.get("pattern")
    normalize_cfg = NormalizeConfig.from_dict(
        anchor.get("normalize"),
        default_remove_whitespace=True,
    )

    target_lines = lines
    if region:
        items_in_region = _filter_items_by_region(lines, region, page_width, page_height)
        target_lines = cluster_lines(items_in_region, y_tol=float(anchor.get("y_tol", 6)))

    for line in target_lines:
        value = line_text(line, normalize=normalize_cfg)
        if pattern and re.search(pattern, value):
            return True
        if text_rule and normalize_text(text_rule, normalize_cfg) in value:
            return True
    return False


def _filter_items_by_region(
    lines: list[Line],
    region: Mapping[str, Any],
    page_width: float,
    page_height: float,
) -> list[OcrItem]:
    x_range = region.get("x") or [0.0, 1.0]
    y_range = region.get("y") or [0.0, 1.0]
    x_min, x_max = float(x_range[0]) * page_width, float(x_range[1]) * page_width
    y_min, y_max = float(y_range[0]) * page_height, float(y_range[1]) * page_height

    items: list[OcrItem] = []
    for line in lines:
        for item in line.items:
            bbox = item.bounding_box
            x_center = (bbox.x1 + bbox.x2) / 2
            y_center = (bbox.y1 + bbox.y2) / 2
            if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
                items.append(item)
    return items


def _page_bounds(items: list[OcrItem]) -> tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for item in items:
        max_x = max(max_x, item.bounding_box.x2)
        max_y = max(max_y, item.bounding_box.y2)
    return max_x, max_y
