"""Marker-backed PDF parsing.

Marker is not imported by the FastAPI app. We run it as a separate Docker job and
commit only the lightweight glue that consumes its output. That keeps the app image
small while preserving the required challenge flow:

    data/in/document.pdf -> Marker artifact -> POST /ingest -> Qdrant -> POST /query -> data/out

The extracted artifacts live under `data/extracted/` and are ignored by git because
they are reproducible from the committed PDF.
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..config import get_settings


_MARKER_PAGE_RE = re.compile(r"^\{(?P<page>\d+)\}-{10,}\s*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_TEXT_BLOCKS = {
    "Caption",
    "Code",
    "Equation",
    "Footnote",
    "ListItem",
    "SectionHeader",
    "Table",
    "Text",
}
_SKIP_BLOCKS = {"PageFooter", "PageHeader", "Picture"}


class _HtmlText(HTMLParser):
    """Small stdlib HTML-to-text helper for Marker JSON fallback."""

    _BLOCK_TAGS = {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "table", "tr"}
    _CELL_TAGS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in self._CELL_TAGS:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS or tag in self._CELL_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return _normalize_text("".join(self.parts))


def extract_pages(pdf_path: Path) -> list[str]:
    """Return 1-indexed PDF pages as a list positionally indexed from zero.

    `/ingest` still receives the PDF from `data/in`. The only extra step is running
    `scripts/run_marker.ps1`, which materializes Marker output from that same PDF.
    """

    markdown_path = _marker_markdown_path(pdf_path)
    if markdown_path.is_file():
        return _pages_from_marker_markdown(markdown_path)

    json_path = _marker_json_path(pdf_path)
    if json_path.is_file():
        return _pages_from_marker_json(json_path)

    raise FileNotFoundError(
        "Marker output is missing. Run '.\\scripts\\run_marker.ps1' first, then call /ingest. "
        f"Expected {markdown_path} or {json_path}."
    )


def _marker_markdown_path(pdf_path: Path) -> Path:
    settings = get_settings()
    return Path(settings.marker_markdown_dir) / pdf_path.stem / f"{pdf_path.stem}.md"


def _marker_json_path(pdf_path: Path) -> Path:
    settings = get_settings()
    return Path(settings.marker_json_dir) / pdf_path.stem / f"{pdf_path.stem}.json"


def _pages_from_marker_markdown(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    pages: dict[int, list[str]] = {}
    current_page: int | None = None

    for line in raw.splitlines():
        marker = _MARKER_PAGE_RE.match(line.strip())
        if marker:
            current_page = int(marker.group("page")) + 1
            pages.setdefault(current_page, [])
            continue
        if current_page is None:
            continue
        cleaned = _clean_marker_markdown_line(line)
        if cleaned:
            pages[current_page].append(cleaned)

    if not pages:
        raise ValueError(f"Marker Markdown had no page separators: {path}")

    return _ordered_pages(pages)


def _pages_from_marker_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: dict[int, list[str]] = {}

    for page in data.get("children", []):
        if page.get("block_type") != "Page":
            continue
        page_no = _page_number(page)
        if page_no is None:
            continue
        blocks = [_text_from_marker_node(child) for child in page.get("children") or []]
        text = "\n\n".join(block for block in blocks if block)
        if text.strip():
            pages.setdefault(page_no, []).append(text)

    if not pages:
        raise ValueError(f"Marker JSON had no readable pages: {path}")

    return _ordered_pages(pages)


def _page_number(node: dict[str, Any]) -> int | None:
    raw_id = str(node.get("id", ""))
    match = re.search(r"/page/(\d+)/", raw_id)
    if match:
        return int(match.group(1)) + 1
    return None


def _text_from_marker_node(node: dict[str, Any]) -> str:
    block_type = str(node.get("block_type", ""))
    if block_type in _SKIP_BLOCKS:
        return ""
    if block_type not in _TEXT_BLOCKS and node.get("children"):
        return _child_text(node)
    if block_type not in _TEXT_BLOCKS:
        return ""

    html_text = str(node.get("html") or "")
    if html_text and "<content-ref" not in html_text:
        return _html_to_text(html_text)

    return _child_text(node)


def _child_text(node: dict[str, Any]) -> str:
    texts: list[str] = []
    for child in node.get("children") or []:
        text = _text_from_marker_node(child)
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _html_to_text(raw_html: str) -> str:
    parser = _HtmlText()
    parser.feed(raw_html)
    return parser.text()


def _clean_marker_markdown_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("![]("):
        return ""
    line = re.sub(r"<span\b[^>]*></span>", "", line)
    line = _MARKDOWN_LINK_RE.sub(r"\1", line)
    line = _HTML_TAG_RE.sub("", line)
    return _normalize_text(html.unescape(line))


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _ordered_pages(pages: dict[int, list[str]]) -> list[str]:
    max_page = max(pages)
    return ["\n\n".join(pages.get(page_no, [])).strip() for page_no in range(1, max_page + 1)]
