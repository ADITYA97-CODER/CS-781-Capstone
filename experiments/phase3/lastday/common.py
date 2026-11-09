"""Last-day (16 Oct) Phase 3 helpers: load gold docs, out-of-fold (OOF) and test probabilities of existing runs,
mean-probability ensembles, the official metric (13-label macro-F1, zero_division=0) and submission validation."""

import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "phase3"))
import rr_data as D  # noqa: E402

LD = os.path.join(ROOT, "outputs", "phase3", "lastday")
CV = os.path.join(ROOT, "runs_p3", "cv")
ALL = os.path.join(ROOT, "runs_p3", "all")
LABELS = D.LABELS
K = len(LABELS)
PO, DIS = D.LAB2ID["PrecedentOverruled"], D.LAB2ID["Dissent"]

# Existing Arch E CV configs: name -> (CV prefix, fixed final epoch, all-data test runs or None)
CONFIGS = {
    "E_ctx_none_nopos": ("ctx_none_nopos", 20, None),
    "E_ctx_none_pos": ("ctx_none_pos", 20, None),
    "E_lorma_win256": ("ctx_win256", 20, None),
    "E_lorma_win1024": ("ctx_win1024", 20, None),
    "E_lorma_mlp_la_1024": ("lorma_mlp_la", 20, None),
    "E_full_mlp_la_1024": ("full_mlp_la", 10, None),           # the 0.5750 test config (CV only, test = e_full_laya_nomarker)
    "E_lorma_win2048": ("lorma_win2048", 20, ["lorma_w2048_s0"]),
    "E_lorma_win3072": ("lorma_win3072", 20, ["lorma_w3072_s0"]),
    "E_full_mlp_la_w2048": ("full_mlp_la_w2048", 10, ["full_mlp_la_w2048_s0"]),
    "E_lorma_mlp_la_w2048": ("lorma_mlp_la_w2048", 20, ["lorma_mlp_la_w2048_s0"]),
}
WIDE3 = ["E_lorma_win3072", "E_lorma_win2048", "E_full_mlp_la_w2048"]   # = outputs/phase3/all_wide3


def gold_docs():
    """All 90 labelled docs (CL/IT train + dev) with text, labels, y."""
    docs = [d for dom in D.DOMAINS for p in ("train", "dev") for d in D.load_split(f"{dom}_{p}")]
    for d in docs:
        d["y"] = np.array([D.LAB2ID[l] for l in d["labels"]])
    return {d["id"]: d for d in docs}


def test_docs():
    return {d["id"]: d for n in ("CL_test", "IT_test") for d in D.load_split(n)}


def _load(path):
    z = np.load(path)
    return {k: np.exp(z[k].astype(np.float32)) for k in z.files}


def oof_probs(prefix, ep, root=CV):
    """OOF probabilities: each doc from the CV half that did not train on it."""
    out = {}
    for f in (0, 1):
        out.update(_load(os.path.join(root, f"{prefix}_f{f}", f"dev_logp_ep{ep}.npz")))
    return out


def test_probs(run_dirs):
    zs = [_load(os.path.join(r, "test_logp.npz")) for r in run_dirs]
    return {k: np.mean([z[k] for z in zs], 0) for k in zs[0]}


def mean_probs(ps, w=None):
    w = np.ones(len(ps)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    return {k: sum(wi * p[k] for wi, p in zip(w, ps)) for k in ps[0]}


def argmax(p):
    return {k: v.argmax(-1) for k, v in p.items()}


def score(gold, pred, ids=None):
    """-> (macro-F1, {label: f1}) over the given docs (default: all gold docs)."""
    ids = sorted(gold) if ids is None else ids
    y = np.concatenate([gold[k]["y"] for k in ids])
    yp = np.concatenate([pred[k] for k in ids])
    f1, _, _, per = D.macro_scores(y, yp)
    return f1, {l: v[2] for l, v in per.items()}


def label_prf(gold, pred, k):
    y = np.concatenate([gold[i]["y"] for i in sorted(gold)])
    yp = np.concatenate([pred[i] for i in sorted(gold)])
    tp, fp, fn = int(((yp == k) & (y == k)).sum()), int(((yp == k) & (y != k)).sum()), int(((yp != k) & (y == k)).sum())
    return tp, fp, fn


def write_and_validate(pred_lab, out_dir):
    """pred_lab: {doc_id: [label strings]} -> out_dir/prediction.json + submission.zip, then re-read the zip and check
    the official rules: ids = CL_test + IT_test exactly, len == len(text) per doc, labels in label_vocab.json,
    prediction.json at the zip root. Sentence order = list order (same as the test file)."""
    os.makedirs(out_dir, exist_ok=True)
    te = test_docs()
    vocab = json.load(open(os.path.join(D.DATA, "Train", "label_vocab.json")))
    path = os.path.join(out_dir, "prediction.json")
    json.dump(pred_lab, open(path, "w"))
    zp = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        got = json.loads(z.read("prediction.json"))
    raw = {}
    for n in ("CL_test", "IT_test"):
        raw.update(json.load(open(os.path.join(D.DATA, "Test", f"{n}.json"))))
    assert set(got) == set(raw), "doc id mismatch"
    for k, sents in raw.items():
        assert isinstance(got[k], list) and len(got[k]) == len(sents), k
        assert all(l in vocab for l in got[k]), k
        assert len(te[k]["text"]) == len(sents)
    return zp


def load_pred_json(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    return json.load(open(path))


def n_changed(a, b):
    return sum(int(x != y) for k in a for x, y in zip(a[k], b[k]))
