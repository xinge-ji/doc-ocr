"""Usage: shared OCR text normalization and line clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.schemas.ocr import OcrItem


@dataclass(frozen=True)
class NormalizeConfig:
    remove_whitespace: bool = False
    remove_brackets: bool = False
    fullwidth_to_halfwidth: bool = False
    lowercase: bool = False

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        default_remove_whitespace: bool = False,
    ) -> "NormalizeConfig":
        if not data:
            return cls(remove_whitespace=default_remove_whitespace)
        return cls(
            remove_whitespace=bool(data.get("remove_whitespace", default_remove_whitespace)),
            remove_brackets=bool(data.get("remove_brackets", False)),
            fullwidth_to_halfwidth=bool(data.get("fullwidth_to_halfwidth", False)),
            lowercase=bool(data.get("lowercase", False)),
        )


@dataclass(frozen=True)
class MergeTokensConfig:
    merge_single_char: bool = False
    max_x_gap: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MergeTokensConfig":
        if not data:
            return cls()
        return cls(
            merge_single_char=bool(data.get("merge_single_char", False)),
            max_x_gap=float(data.get("max_x_gap", 0.0)),
        )


@dataclass(frozen=True)
class TextToken:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    page: int

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class Line:
    items: list[OcrItem]
    y_center: float
    page: int

    def sorted_items(self) -> list[OcrItem]:
        return sorted(self.items, key=lambda item: item.bounding_box.x1)


def normalize_text(text: str, config: NormalizeConfig) -> str:
    value = text
    if config.fullwidth_to_halfwidth:
        value = _to_halfwidth(value)
    if config.remove_brackets:
        value = _strip_brackets(value)
    if config.remove_whitespace:
        value = "".join(value.split())
    if config.lowercase:
        value = value.lower()
    return value


def cluster_lines(items: Iterable[OcrItem], *, y_tol: float) -> list[Line]:
    sorted_items = sorted(
        items,
        key=lambda item: (item.page, (item.bounding_box.y1 + item.bounding_box.y2) / 2),
    )
    lines: list[Line] = []
    for item in sorted_items:
        y_center = (item.bounding_box.y1 + item.bounding_box.y2) / 2
        target_line: Line | None = None
        for line in reversed(lines):
            if line.page != item.page:
                break
            if abs(line.y_center - y_center) <= y_tol:
                target_line = line
                break
        if target_line is None:
            lines.append(Line(items=[item], y_center=y_center, page=item.page))
        else:
            target_line.items.append(item)
            target_line.y_center = (target_line.y_center + y_center) / 2
    return lines


def to_tokens(items: Sequence[OcrItem]) -> list[TextToken]:
    tokens: list[TextToken] = []
    for item in items:
        bbox = item.bounding_box
        tokens.append(
            TextToken(
                text=item.text or "",
                x1=bbox.x1,
                y1=bbox.y1,
                x2=bbox.x2,
                y2=bbox.y2,
                page=item.page,
            )
        )
    return tokens


def merge_tokens(tokens: Sequence[TextToken], config: MergeTokensConfig) -> list[TextToken]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: token.x1)
    merged: list[TextToken] = [ordered[0]]
    for token in ordered[1:]:
        prev = merged[-1]
        gap = token.x1 - prev.x2
        if config.max_x_gap > 0:
            should_merge = gap <= config.max_x_gap
        else:
            should_merge = False

        if config.merge_single_char and config.max_x_gap > 0:
            should_merge = should_merge and (
                len(prev.text.strip()) <= 1 or len(token.text.strip()) <= 1
            )
        if should_merge:
            merged[-1] = TextToken(
                text=f"{prev.text}{token.text}",
                x1=min(prev.x1, token.x1),
                y1=min(prev.y1, token.y1),
                x2=max(prev.x2, token.x2),
                y2=max(prev.y2, token.y2),
                page=prev.page,
            )
        else:
            merged.append(token)
    return merged


def join_tokens(tokens: Sequence[TextToken], *, normalize: NormalizeConfig | None = None) -> str:
    text = "".join(token.text for token in tokens)
    if normalize is None:
        return text
    return normalize_text(text, normalize)


def line_text(line: Line, *, normalize: NormalizeConfig | None = None) -> str:
    items = line.sorted_items()
    tokens = to_tokens(items)
    return join_tokens(tokens, normalize=normalize)


def _to_halfwidth(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            result.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return "".join(result)


def _strip_brackets(text: str) -> str:
    brackets = {"(", ")", "（", "）", "[", "]", "【", "】"}
    return "".join(char for char in text if char not in brackets)
