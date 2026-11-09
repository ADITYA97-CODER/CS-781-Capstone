"""Count outputs that contain a court header or a closing formula.

  python phase1/diagnostics/output_formulas.py name=path/prediction.json [name=path ...]

Headers begin a document and closing formulas end one, so a continuation containing them has
likely run past the end of its document into a new one (or ended it early). The local
references are included (as 'ref') when the prediction ids match them, as the natural rate.
"""

import json
import os
import re
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_REF = os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_ref.json")

HEADERS = re.compile(r"IN THE (SUPREME|HIGH) COURT|\bREPORTABLE\b|(CIVIL|CRIMINAL) (APPELLATE|ORIGINAL) JURISDICTION")
CLOSERS = re.compile(r"\b(appeal is (dis|al)lowed|petition is (dis|al)lowed|appeal dismissed|appeal allowed|"
                     r"no order as to costs|rule (is )?(made absolute|discharged)|parties to bear|"
                     r"stands? disposed of|bail bonds)", re.I)

runs = dict(a.split("=", 1) for a in sys.argv[1:])
if not runs:
    sys.exit(__doc__)
preds = {n: json.load(open(p)) for n, p in runs.items()}
ref = json.load(open(LOCAL_REF))
if all(set(p) == set(ref) for p in preds.values()):
    preds = {"ref": ref, **preds}

print(f"{'run':24s} {'n':>5s} {'court header':>13s} {'closing formula':>16s}")
for name, p in preds.items():
    head = [k for k, v in p.items() if HEADERS.search(v["text"])]
    close = [k for k, v in p.items() if CLOSERS.search(v["text"])]
    print(f"{name:24s} {len(p):>5d} {len(head):>13d} {len(close):>16d}")
    if name != "ref":
        print(f"{'':24s} header ids: {head[:12]}\n{'':24s} closing ids: {close[:12]}")
