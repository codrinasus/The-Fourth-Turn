#!/usr/bin/env python
"""Run app/rag/chunking.py over the parsed pages and write the chunks to disk.

    uv run python scripts/dump_chunks.py [--view text|markdown] [--out data/chunks]

This calls the *same* `chunk_pages()` that `/ingest` uses (app/rag/chunking.py), so what
you read here is exactly what would be embedded — but without Qdrant or the embedding
model, so the loop is instant. Edit chunking.py, re-run, look at the files.

Writes one file per chunk, so you can open, diff and grep them individually:

    data/chunks/chunk-0000_page-001.txt   # the chunk text, nothing else
    data/chunks/index.json                # stats + {index, page, chars, words, file}

The output folder is rebuilt on every run (stale chunk files are removed first), so it
always reflects the current chunking.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.chunking import chunk_pages

PARSED_DEFAULT = Path("data/parsed")
CHUNKS_DEFAULT = Path("data/chunks")


def load_pages(parsed_dir: Path, view: str) -> tuple[list[str], dict]:
    """Page texts from data/parsed/document.json, falling back to parsing the PDF."""
    doc_json = parsed_dir / "document.json"
    if doc_json.is_file():
        doc = json.loads(doc_json.read_text(encoding="utf-8"))
        return [p[view] for p in doc["pages"]], doc

    from parse_pdf import parse_pdf  # same folder

    pdfs = sorted(Path("data/in").glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no {doc_json} and no *.pdf in data/in/ — run scripts/parse_pdf.py first")
    doc = parse_pdf(pdfs[0])
    return [p[view] for p in doc["pages"]], doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--view",
        choices=["text", "markdown", "text_raw"],
        default="text",
        help="which page view to feed the chunker (default: text)",
    )
    ap.add_argument("--parsed", type=Path, default=PARSED_DEFAULT)
    ap.add_argument("--out", type=Path, default=CHUNKS_DEFAULT)
    args = ap.parse_args()

    pages, doc = load_pages(args.parsed, args.view)
    chunks = chunk_pages(pages)
    if not chunks:
        raise SystemExit("chunk_pages() returned nothing")

    records = [
        {
            "index": c.index,
            "page": c.page,
            "chars": len(c.text),
            "words": len(c.text.split()),
            "file": f"chunk-{c.index:04d}_page-{c.page:03d}.txt",
            "text": c.text,
        }
        for c in chunks
    ]
    sizes = [r["chars"] for r in records]
    stats = {
        "source": doc.get("source", ""),
        "view": args.view,
        "pages": len(pages),
        "chunks": len(records),
        "chars_min": min(sizes),
        "chars_median": int(statistics.median(sizes)),
        "chars_max": max(sizes),
        "pages_covered": len({r["page"] for r in records}),
    }

    # Rebuild the folder: a new chunking usually yields a different count, and leftover
    # files from the previous run would silently look like current chunks.
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("chunk-*.txt"):
        stale.unlink()
    for r in records:
        (args.out / r["file"]).write_text(r["text"] + "\n", encoding="utf-8")

    index = {
        "stats": stats,
        "chunks": [{k: v for k, v in r.items() if k != "text"} for r in records],
    }
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{stats['chunks']} chunks from {stats['pages']} pages (view={args.view})")
    print(
        f"  chars  min {stats['chars_min']}  median {stats['chars_median']}  max {stats['chars_max']}"
    )
    print(f"  pages covered: {stats['pages_covered']}/{stats['pages']}")
    print(f"  -> {args.out}/chunk-*.txt  +  {args.out / 'index.json'}")

    over = [r["index"] for r in records if r["chars"] > 2000]
    if over:
        print(f"  note: {len(over)} chunk(s) over 2000 chars — likely truncated by the embedder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
