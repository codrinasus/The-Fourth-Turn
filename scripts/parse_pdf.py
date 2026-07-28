#!/usr/bin/env python
"""Parse the PDF in data/in/ into data/parsed/ — a stable, on-disk parse artifact.

    uv run python scripts/parse_pdf.py [--pdf data/in/document.pdf] [--out data/parsed]

Why this exists: everything downstream (chunking, retrieval, prompting) only needs
*pages of text*. Writing that parse to disk once decouples it from whichever extractor
we end up using — swap pypdf for PyMuPDF/Docling/GROBID later, re-run this script, and
nothing downstream changes as long as the JSON schema below stays the same.

Output (schema_version 2):

    data/parsed/
      document.json      # the whole parse: metadata + one record per page
      document.md        # markdown, with <!-- page N --> markers between pages
      pages/page-001.txt # one normalised page per file (easy to grep / eyeball)

Each page record carries three views of the same page, on purpose:
  text_raw — exactly what the extractor returned. Use this to *verify* a Source.quote,
             because it is the closest thing we have to "what is on the page".
  text     — running header/footer removed, line-break hyphenation joined, whitespace
             normalised. Use this for embedding and for building quotes; it reads like
             prose instead of like a column of ragged lines.
  markdown — `text` reflowed into paragraphs with `#` headings, italic figure/table
             captions and list items. Use this when chunking structure-aware (split on
             headings, keep the section title on the chunk).

Header/footer removal is learned from the document (lines that repeat across most pages),
not hardcoded for this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

SCHEMA_VERSION = 2
EXTRACTOR = "pypdf"

# Fraction of pages a first/last line must repeat on to count as a running header/footer.
_REPEAT_THRESHOLD = 0.4
# "1.2.3 SOME HEADING" / "4 INTRODUCTION" on a line of its own.
_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Za-z][^.]{2,90})$")
# The same, but set run-in: "3.1.2 Some title. Body text continues on the same line."
_RUNIN_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){1,2})\s+([A-Za-z][^.]{2,90})\.\s+(\S.*)$")
_CAPTION_RE = re.compile(
    r"^(Fig\.|Figure|Table|Alg\.|Algorithm|Listing)\s*\d+[.:]?\s", re.IGNORECASE
)
_BULLET_RE = re.compile(r"^\s*([-–—•*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+\S")
# A body line much shorter than the column width ends its paragraph.
_SHORT_LINE_RATIO = 0.78
_WEIRD_SPACE_RE = re.compile(r"[     \u200b]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-‐‑]\n(\w)")


def _digit_mask(line: str) -> str:
    """Running headers differ only by the page number — mask digits before comparing."""
    return re.sub(r"\d+", "#", line.strip())


def _find_running_lines(pages: list[str]) -> tuple[set[str], set[str]]:
    """Learn which first/last lines are boilerplate, as masked patterns."""
    firsts: Counter[str] = Counter()
    lasts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        firsts[_digit_mask(lines[0])] += 1
        lasts[_digit_mask(lines[-1])] += 1

    n = max(1, sum(1 for p in pages if p.strip()))
    cutoff = max(2, int(n * _REPEAT_THRESHOLD))
    return (
        {pat for pat, c in firsts.items() if c >= cutoff},
        {pat for pat, c in lasts.items() if c >= cutoff},
    )


def _normalise(text: str, headers: set[str], footers: set[str]) -> tuple[str, list[str]]:
    """Strip boilerplate, join hyphenated line breaks, tidy whitespace.

    Returns the cleaned text and the boilerplate lines that were dropped.
    """
    lines = text.splitlines()
    # Only the outermost non-empty lines are candidates — never touch the body.
    dropped: list[str] = []
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _digit_mask(lines[0]) in headers:
        dropped.append(lines.pop(0).strip())
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _digit_mask(lines[-1]) in footers:
        dropped.append(lines.pop().strip())

    body = "\n".join(_WEIRD_SPACE_RE.sub(" ", ln).rstrip() for ln in lines)
    body = _HYPHEN_BREAK_RE.sub(r"\1\2", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip(), dropped


def _headings(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            out.append(f"{m.group(1)} {m.group(2).strip()}")
    return out


def _heading_md(number: str, title: str) -> str:
    """`2` -> `##`, `2.1` -> `###`, `2.1.3` -> `####` (h1 is reserved for the title)."""
    level = min(6, 2 + number.count("."))
    return f"{'#' * level} {number} {title.strip()}"


def _column_width(pages: list[str]) -> int:
    """Typical full line length, used to spot paragraph-final (short) lines."""
    lengths = sorted(len(ln) for text in pages for ln in text.splitlines() if ln.strip())
    if not lengths:
        return 0
    return lengths[int(0.75 * (len(lengths) - 1))]


def to_markdown(text: str, column_width: int) -> str:
    """Reflow one normalised page into markdown.

    PDF text arrives as hard-wrapped lines with no paragraph markers, so paragraphs are
    inferred: a body line shorter than ~78% of the column width ends the paragraph.
    Headings, captions and list items are emitted on their own. Everything is per page —
    a paragraph split across a page break stays split, so page provenance never blurs.
    """
    short = column_width * _SHORT_LINE_RATIO
    out: list[str] = []
    para: list[str] = []
    style = ""  # markdown wrapper for the paragraph being built: "", "*" or "- "

    def flush() -> None:
        nonlocal style
        if para:
            block = " ".join(para)
            out.append(f"*{block}*" if style == "*" else f"{style}{block}")
            para.clear()
        style = ""

    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if not line:
            flush()
            continue

        m = _HEADING_RE.match(line)
        if m:
            flush()
            out.append(_heading_md(m.group(1), m.group(2)))
            continue

        m = _RUNIN_HEADING_RE.match(line)
        if m and not para:  # only when it starts a block, not mid-paragraph
            out.append(_heading_md(m.group(1), m.group(2)))
            para.append(m.group(3))
            continue

        # Captions and list items open a new block that keeps its continuation lines.
        if _CAPTION_RE.match(line):
            flush()
            style = "*"
        elif _BULLET_RE.match(line):
            flush()
            style = "- "
            line = line.lstrip("-–—•* ")

        para.append(line)
        # Short line = end of paragraph. Unless its neighbours are short too: that is a
        # table/list-of-values block, where every line stands on its own anyway.
        if len(line) < short:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not (nxt and len(nxt) < short):
                flush()

    flush()
    md = "\n\n".join(block for block in out if block.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def parse_pdf(pdf: Path) -> dict:
    reader = PdfReader(str(pdf))
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    headers, footers = _find_running_lines(raw_pages)
    column_width = _column_width(raw_pages)

    pages: list[dict] = []
    current_section = ""
    for page_no, raw in enumerate(raw_pages, start=1):
        text, dropped = _normalise(raw, headers, footers)
        found = _headings(text)
        # The section a page *starts* in is the last heading seen before it.
        section_at_start = current_section
        if found:
            current_section = found[-1]
        pages.append(
            {
                "page": page_no,
                "text": text,
                "text_raw": raw,
                "markdown": to_markdown(text, column_width),
                "chars": len(text),
                "words": len(text.split()),
                "section": section_at_start,
                "headings": found,
                "dropped_boilerplate": dropped,
                "empty": not text.strip(),
            }
        )

    meta = reader.metadata or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "source": pdf.name,
        "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "extractor": f"{EXTRACTOR} ({_pypdf_version()})",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_count": len(pages),
        "column_width": column_width,
        "title": str(meta.get("/Title", "") or ""),
        "author": str(meta.get("/Author", "") or ""),
        "running_headers": sorted(headers),
        "running_footers": sorted(footers),
        "pages": pages,
    }


def _pypdf_version() -> str:
    try:
        import pypdf

        return pypdf.__version__
    except (ImportError, AttributeError):  # informational only
        return "unknown"


def write_outputs(doc: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "document.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    for stale in pages_dir.glob("page-*.txt"):
        stale.unlink()
    for p in doc["pages"]:
        (pages_dir / f"page-{p['page']:03d}.txt").write_text(p["text"] + "\n", encoding="utf-8")

    md = [f"# {doc['title'] or doc['source']}", ""]
    for p in doc["pages"]:
        md.append(f"<!-- page {p['page']} -->")
        md.append("")
        md.append(p["markdown"])
        md.append("")
    (out_dir / "document.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", type=Path, default=None, help="defaults to the first data/in/*.pdf")
    ap.add_argument("--out", type=Path, default=Path("data/parsed"))
    args = ap.parse_args()

    pdf = args.pdf
    if pdf is None:
        candidates = sorted(Path("data/in").glob("*.pdf"))
        if not candidates:
            print("no *.pdf in data/in/", file=sys.stderr)
            return 1
        pdf = candidates[0]

    doc = parse_pdf(pdf)
    write_outputs(doc, args.out)

    empty = [p["page"] for p in doc["pages"] if p["empty"]]
    print(f"parsed {pdf} -> {args.out}")
    print(f"  pages: {doc['page_count']}  empty: {len(empty)} {empty if empty else ''}")
    print(f"  words: {sum(p['words'] for p in doc['pages'])}")
    print(f"  dropped headers: {doc['running_headers']}")
    print(f"  dropped footers: {doc['running_footers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
