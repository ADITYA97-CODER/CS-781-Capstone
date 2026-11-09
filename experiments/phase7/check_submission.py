"""Validate a PCR submission dir: submission.zip holds exactly prediction.json; 237 test ids; 7 unique pool ids per
query, query itself excluded. If dev_oof_prediction.json exists, score it with the official evaluator.
  python check_submission.py <dir> [<dir> ...]"""

import json
import os
import subprocess
import sys
import zipfile

import pcr_data as D

EVAL = os.path.join(D.ROOT, "official_eval", "Evaluation Script", "evaluate_phase7.py")
PY = sys.executable


def check(d):
    z = zipfile.ZipFile(os.path.join(d, "submission.zip"))
    assert z.namelist() == ["prediction.json"], z.namelist()
    pred = json.loads(z.read("prediction.json"))
    qs, _, pool = D.load("test")
    assert set(pred) == set(qs), "test ids"
    for q, v in pred.items():
        assert isinstance(v, list) and len(v) == 7 and len(set(v)) == 7, q
        assert all(isinstance(c, str) and c in pool for c in v), q
        assert q not in v, q
    msg = f"{d}: OK ({len(pred)} queries)"
    # official validator on test structure (truth = test ids with empty gold lists; only validation matters)
    tmp = os.path.join(d, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    json.dump({q: [] for q in qs}, open(os.path.join(tmp, "truth.json"), "w"))
    json.dump(list(pool), open(os.path.join(tmp, "cands.json"), "w"))
    json.dump(pred, open(os.path.join(tmp, "prediction.json"), "w"))
    r = subprocess.run([PY, EVAL, "--truth", os.path.join(tmp, "truth.json"), "--prediction",
                        os.path.join(tmp, "prediction.json"), "--candidate-ids", os.path.join(tmp, "cands.json")],
                       capture_output=True, text=True)
    msg += " | official validator: " + ("passed" if "Validation passed" in r.stdout else "FAILED " + r.stdout[-300:])
    oof = os.path.join(d, "dev_oof_prediction.json")
    if os.path.exists(oof):
        r = subprocess.run([PY, EVAL, "--truth", os.path.join(D.DATA, "Dev", "dev.json"), "--prediction", oof,
                            "--candidate-ids", os.path.join(d, "dev_candidate_ids.json")], capture_output=True, text=True)
        f1 = [l for l in r.stdout.splitlines() if l.startswith("Micro-F1")]
        msg += f" | dev OOF official {f1[0] if f1 else r.stdout[-300:]}"
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(msg, flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        check(d)
