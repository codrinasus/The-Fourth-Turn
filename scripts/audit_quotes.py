"""Check every submitted evidence quote against the real PDF.

    uv run python scripts/audit_quotes.py            # audit submission/
    uv run python scripts/audit_quotes.py data/out   # audit any directory of responses

For each `Source` in each response it asks the one question grading asks: **does this
exact text appear on that exact page of `data/in/document.pdf`?** The comparison is done
against pypdf's extraction — a different parser from the Docling one we index with, so a
pass means two independent readers agree the text is there.

Whitespace is normalised on both sides and nothing else is: no case folding, no
punctuation folding. A quote that only matches with the dashes and apostrophes smoothed
over is reported as a FAIL, because that is precisely the failure `app/rag/verbatim.py`
exists to prevent and we would rather see it than hide it.

Exit code is non-zero if anything fails, so it can gate a push.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.verbatim import is_verbatim


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "submission")
    files = sorted(root.rglob("*.json"))

    total = ok = empty = 0
    failures: list[str] = []

    for path in files:
        if path.name == "team.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"{path}: not valid JSON")
            continue
        if not data or not data.get("answer"):
            empty += 1
            failures.append(f"{path}: empty — this question would score 0")
            continue

        for i, source in enumerate(data.get("sources", []), start=1):
            total += 1
            quote, page = source.get("quote", ""), source.get("page", 0)
            if is_verbatim(quote, page):
                ok += 1
            else:
                failures.append(f"{path.name} source {i} (page {page}): {quote[:90]!r}")

    rate = f"{ok}/{total}" + (f" ({100 * ok / total:.0f}%)" if total else "")
    print(f"quotes verbatim on the cited page: {rate}")
    print(f"empty answer files: {empty}")
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
