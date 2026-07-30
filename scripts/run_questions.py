"""Ask the nine chosen questions and file the answers into `submission/`.

    uv run python scripts/run_questions.py            # all nine
    uv run python scripts/run_questions.py 2          # just level 2

Questions come from `questions/chosen.json` — the same nine as the Postman collection.
Each level is sent **in order** so the level-2 follow-ups see the earlier turns, and the
raw `POST /query` response is written verbatim to `submission/level-<n>/q<i>.json`. Nothing
is hand-edited on the way: what the pipeline returned is what gets submitted.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "http://localhost:8791/query"


def ask(question: str, level: int) -> dict:
    body = json.dumps({"question": question, "level": level}).encode()
    req = urllib.request.Request(APP, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main() -> int:
    wanted = {int(a) for a in sys.argv[1:]} or {1, 2, 3}
    questions = json.loads((ROOT / "questions" / "chosen.json").read_text())

    for q in questions:
        if q["level"] not in wanted:
            continue
        started = time.perf_counter()
        try:
            resp = ask(q["question"], q["level"])
        except Exception as e:  # noqa: BLE001 - a failed question must not stop the rest
            print(f"{q['id']}  FAILED  {e}")
            continue
        out = ROOT / "submission" / f"level-{q['level']}" / f"{q['id']}.json"
        out.write_text(json.dumps(resp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        took = time.perf_counter() - started
        pages = ", ".join(f"p{s['page']}:{s['score']:.2f}" for s in resp["sources"])
        print(f"{q['id']}  {took:6.1f}s  [{pages}]")
        print(f"    {resp['answer'][:200].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
