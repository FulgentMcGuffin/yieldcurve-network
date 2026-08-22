"""Repair truncated ycs_multilayer.ipynb (missing footer + broken last cell)."""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_FOOTER = """ "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}"""


def main() -> None:
    path = Path("src/ycn/ycs_multilayer.ipynb")
    text = path.read_text(encoding="utf-8")
    print(f"Original size: {len(text):,} chars")

    cell_pat = re.compile(r'\n  \{\n   "cell_type":')
    matches = list(cell_pat.finditer(text))
    print(f"Found {len(matches)} cell starts")

    header = text[: matches[0].start() + 1]

    def build(n: int) -> str | None:
        if n <= 0:
            return None
        start = matches[0].start() + 1
        end = matches[n].start() if n < len(matches) else len(text)
        cell_block = text[start:end].rstrip().rstrip(",")
        return header + cell_block + "\n ],\n" + DEFAULT_FOOTER

    best = 0
    for n in range(1, len(matches)):
        repaired = build(n)
        assert repaired is not None
        try:
            json.loads(repaired)
            best = n
        except json.JSONDecodeError as e:
            print(f"n={n} FAIL: {e.msg} @ {e.pos}")
            break

    if best == 0:
        raise SystemExit("Could not repair notebook")

    print(f"Keeping {best} complete cells (dropped {len(matches) - best} truncated)")
    nb = json.loads(build(best))

    last = nb["cells"][-1]
    if last.get("outputs"):
        print("Clearing outputs on last retained cell")
        last["outputs"] = []
        last["execution_count"] = None

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = [src]
        cell["source"] = [
            line.replace("from tgraphportfolio.analysis.", "from ycn.analysis.")
            for line in src
        ]

    backup = path.with_suffix(".ipynb.corrupted.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup saved: {backup}")

    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    json.load(open(path, encoding="utf-8"))
    print(f"Repaired notebook: {len(nb['cells'])} cells, {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
