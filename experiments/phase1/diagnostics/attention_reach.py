"""Do Arch D's sliding-window (512) layers hurt reuse of specifics from far back in the prompt?

For every "specific" (citation, section, article, rule, Act+year, witness id, date, amount,
named entity) that appears in the prompt AND in the reference continuation:
  distance = legal_bpe_32k tokens from the start of its LAST prompt occurrence to the prompt end
  hit      = the specific also appears in the model output
Reuse rate is reported per distance bucket with prompt-level (cluster) bootstrap 95% CIs.

Only the prompt tokens the model actually saw count: generate.py cuts prompts to 1216 tokens and
left-truncates each batch of 128 (sorted by length) to its shortest member. Specifics whose last
occurrence lies in the dropped part are excluded (and counted).

The Llama baseline (full attention in every layer) is analysed the same way as a control curve.

Usage:
  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/attention_reach.py            # submitted Arch D MBR outputs
  CUDA_VISIBLE_DEVICES=0 python phase1/diagnostics/attention_reach.py --pred <prediction.json> --out <dir> \
      --label "Arch E" [--gen-batch-size 128 --max-new-tokens 320 --max-seq-len 1536]
The --gen-* settings must be those generate.py used, so the visible prompt is reconstructed exactly.
phase1/diagnostics/eval_checkpoint.py runs generation + MBR + this script for any checkpoint.
"""

import argparse
import bisect
import json
import os
import re
from collections import defaultdict

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
BUCKETS = [0, 128, 256, 384, 512, 768, 1024, 10 ** 9]
BUCKET_NAMES = ["0-128", "128-256", "256-384", "384-512", "512-768", "768-1024", "1024+"]
WINDOW = 512

# ----------------------------------------------------------------------------- regex specifics

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MON = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
NUMLIST = r"(\d+[A-Z]{0,2}(?:\s*(?:,|and|&|/|or)\s*\d+[A-Z]{0,2})*)"
ACT_WORD = r"(?:[A-Z][\w.'&()-]*|of|the|and|for|in|on|to|&)"


def _date(d, m, y):
    d, y = int(d), int(y)
    if len(str(y)) == 2:
        y += 1900 if y > 25 else 2000
    m = MONTHS[m[:3].lower()] if isinstance(m, str) and not m.isdigit() else int(m)
    if not (1 <= d <= 31 and 1 <= m <= 12 and 1850 <= y <= 2030):
        return None
    return [f"date:{y:04d}-{m:02d}-{d:02d}"]


def _nums(prefix, s):
    return [f"{prefix}:{n.upper()}" for n in re.findall(r"\d+[A-Za-z]{0,2}", s)]


def _act_name(s):
    s = re.sub(r"\s+", " ", s.lower())
    while re.match(r"(the|of|and|in|on|to|for|&) ", s):
        s = s.split(" ", 1)[1]
    return s


def _court(s):
    return re.sub(r"[^a-z]", "", s.lower())


PATTERNS = [
    ("citation", r"\(\s*(\d{4})\s*\)\s*(\d+)\s*S\.?\s?C\.?\s?C\.?\s*(\d+)",
     lambda m: [f"cit:scc {m[1]} {m[2]} {m[3]}"]),
    ("citation", r"\b(\d{4})\s*\(\s*(\d+)\s*\)\s*S\.?\s?C\.?\s?C\.?\s*(\d+)",
     lambda m: [f"cit:scc {m[1]} {m[2]} {m[3]}"]),
    ("citation", r"\bA\.?\s?I\.?\s?R\.?\s*\(?\s*(\d{4})\s*\)?\s*([A-Z][A-Za-z.]*(?:\s?[A-Z][a-z.]*)?)\s*(\d+)",
     lambda m: [f"cit:air {m[1]} {_court(m[2])} {m[3]}"]),
    ("citation", r"\[\s*(\d{4})\s*\]\s*(\d+)?\s*S\.?\s?C\.?\s?R\.?\s*(\d+)",
     lambda m: [f"cit:scr {m[1]} {m[2] or ''} {m[3]}"]),
    ("citation", r"\b(\d{4})\s+(Cri\.?\s?L\.?\s?J\.?|S\.?C\.?C\.?\s?\(Cri\)|All\.?\s?L\.?\s?J\.?|Bom\.?\s?L\.?\s?R\.?|Mad\.?\s?L\.?\s?J\.?)\s*(\d+)",
     lambda m: [f"cit:{_court(m[2])} {m[1]} {m[3]}"]),
    ("section", r"(?i:\b(?:sections?|secs?\.?|ss?\.|u/ss?\.?))\s*" + NUMLIST, lambda m: _nums("sec", m[1])),
    ("article", r"(?i:\b(?:articles?|arts?\.))\s*" + NUMLIST, lambda m: _nums("art", m[1])),
    ("rule", r"(?i:\brules?)\s+" + NUMLIST, lambda m: _nums("rule", m[1])),
    ("act", r"\b(" + ACT_WORD + r"(?:\s+" + ACT_WORD + r"){0,10}?\s+(?:Act|Code|Ordinance)(?:\s+of(?:\s+[A-Z][\w.]*)+)?)\s*,?\s*(\d{4})\b",
     lambda m: [f"act:{_act_name(m[1])} {m[2]}"]),
    ("witness", r"\b([PDC])\.?\s?W\.?\s*[-.:]?\s*(\d{1,3})\b", lambda m: [f"wit:{m[1]}W-{int(m[2])}"]),
    ("date", r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:day\s+of\s+|of\s+)?" + MON + r"\.?,?\s*(\d{4})\b",
     lambda m: _date(m[1], m[2], m[3])),
    ("date", r"\b" + MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", lambda m: _date(m[2], m[1], m[3])),
    ("date", r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", lambda m: _date(m[1], m[2], m[3])),
    ("date", r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b", lambda m: _date(m[1], m[2], m[3])),
    ("amount", r"(?:\bRs\.?|₹|(?i:\brupees))\s*(\d[\d,]*(?:\.\d+)?)",
     lambda m: [f"amt:{re.sub(r'\.0+$', '', m[1].replace(',', ''))}"] if m[1].replace(',', '').strip('.') else None),
]
PATTERNS = [(t, re.compile(p), f) for t, p, f in PATTERNS]


def regex_specifics(text):
    """-> {key: (type, [char start of every occurrence])}"""
    out = {}
    for typ, pat, fn in PATTERNS:
        for m in pat.finditer(text):
            keys = fn(m)
            for k in keys or []:
                out.setdefault(k, (typ, []))[1].append(m.start())
    return out


# ----------------------------------------------------------------------------- NER specifics

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC"}
GENERIC = {
    "court", "high court", "supreme court", "hon'ble court", "honble court", "hon'ble supreme court",
    "hon'ble high court", "tribunal", "state", "government", "state government", "central government",
    "union of india", "india", "respondent", "respondents", "appellant", "appellants", "petitioner",
    "petitioners", "section", "article", "act", "code", "ipc", "crpc", "cr.p.c", "cpc", "c.p.c",
    "i.p.c", "the act", "bench", "division bench", "division bench of this court", "this court",
    "the high court", "the supreme court", "the tribunal", "government of india", "sessions court",
    "trial court", "the trial court", "learned counsel", "counsel", "accused", "the state", "the court",
    "commissioner", "collector", "the government", "magistrate", "police", "police station",
    "constitution", "the constitution", "parliament", "legislature", "appellate tribunal",
}


def ner_norm(s):
    s = re.sub(r"\s+", " ", s.strip().strip(".,;:()'\"-")).lower()
    s = re.sub(r"^(the|of|and|in|mr\.?|mrs\.?|ms\.?|dr\.?|shri|smt\.?|sri) ", "", s)
    return s


def ner_ok(ent):
    k = ner_norm(ent.text)
    return (ent.label_ in NER_LABELS and len(k) >= 4 and not re.search(r"\d", k) and k not in GENERIC
            and any(c.isupper() for c in ent.text) and len(k.split()) <= 8)


def ner_pattern(k):
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, k.split())) + r"(?!\w)", re.I)


# ----------------------------------------------------------------------------- per-example analysis

def visible_lengths(prompts, tok, max_prompt=1536 - 320, batch=128):
    """generate.py: cut to max_prompt tokens, sort by length, left-truncate each batch to its shortest."""
    lens = {k: min(len(tok(v["text"], add_special_tokens=False)["input_ids"]), max_prompt) for k, v in prompts.items()}
    order = sorted(lens, key=lambda k: lens[k])
    vis = {}
    for b in range(0, len(order), batch):
        keys = order[b:b + batch]
        L = min(lens[k] for k in keys)
        for k in keys:
            vis[k] = L
    return vis


def visible_starts_llama(prompts, llama_tok, max_len=1536 - 384):
    """llama_sample.py: left truncation to 1152 Llama tokens, left padding (no batch truncation).
    Returns the character index where the visible text starts."""
    out = {}
    for k, v in prompts.items():
        enc = llama_tok(v["text"], add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        # the model sees BOS + the last (max_len - 1) tokens
        out[k] = 0 if len(offs) <= max_len - 1 else offs[len(offs) - (max_len - 1)][0]
    return out


def bucket_of(d):
    return bisect.bisect_right(BUCKETS, d) - 1


def cluster_bootstrap(hits, tots, n_boot=2000, seed=0):
    """hits, tots: (n_prompts, n_buckets). Resample prompts -> 95% CI of sum(hits)/sum(tots) per bucket."""
    rng = np.random.default_rng(seed)
    n = hits.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    h, t = hits[idx].sum(1), tots[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = h / t
    return np.nanpercentile(r, 2.5, axis=0), np.nanpercentile(r, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=P("outputs/phase1/mbr/mbr_archD4.json"))
    ap.add_argument("--llama-pred", default="", help="optional Llama control, e.g. outputs/phase1/mbr/mbr_llama4.json")
    ap.add_argument("--out", default=P("outputs/phase1/attention_diagnostics"))
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--label", default="Arch D (hybrid, window 512)")
    ap.add_argument("--gen-batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts/tokenizer/legal_bpe_32k"))
    prompts = json.load(open(P("data/corpus/eval/local_test.json")))
    refs = json.load(open(P("data/corpus/eval/local_ref.json")))
    systems = {"model": json.load(open(args.pred))}
    if args.llama_pred and os.path.exists(args.llama_pred):
        systems["llama"] = json.load(open(args.llama_pred))
    ids = list(prompts)

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        nlp.max_length = 2_000_000
        print("spaCy NER: en_core_web_sm")
    except Exception as e:           # NER is optional
        print("spaCy NER unavailable, regex only:", e)
        nlp = None

    # --- token geometry of each prompt (legal_bpe_32k)
    enc = {k: tok(prompts[k]["text"], add_special_tokens=False, return_offsets_mapping=True) for k in ids}
    n_tok = {k: len(enc[k]["input_ids"]) for k in ids}
    tok_starts = {k: [o[0] for o in enc[k]["offset_mapping"]] for k in ids}
    vis_len = visible_lengths(prompts, tok, args.max_seq_len - args.max_new_tokens, args.gen_batch_size)
    vis_char = {"model": {k: (tok_starts[k][n_tok[k] - vis_len[k]] if vis_len[k] < n_tok[k] else 0) for k in ids}}
    if "llama" in systems:
        ltok = AutoTokenizer.from_pretrained(P("outputs/llama_phase1/legal-llama-final"))
        vis_char["llama"] = visible_starts_llama(prompts, ltok)

    def tok_distance(k, char_pos):
        j = bisect.bisect_right(tok_starts[k], char_pos) - 1
        return n_tok[k] - j

    # --- NER over prompts, references, outputs
    texts = {("prompt", k): prompts[k]["text"] for k in ids}
    texts.update({("ref", k): refs[k]["text"] for k in ids})
    for s, pred in systems.items():
        texts.update({(s, k): pred[k]["text"] for k in ids})
    ner = defaultdict(dict)                       # (src, k) -> {norm: label}
    if nlp is not None:
        keys = list(texts)
        for key, doc in zip(keys, nlp.pipe((texts[x] for x in keys), batch_size=32, n_process=8)):
            for e in doc.ents:
                if ner_ok(e):
                    ner[key][ner_norm(e.text)] = e.label_
        print("NER done")

    def specifics(src, k):
        """{key: (type, positions)} for a text: regex keys + NER keys located by string search."""
        text = texts[(src, k)]
        sp = regex_specifics(text)
        for nk, lab in ner[(src, k)].items():
            pos = [m.start() for m in ner_pattern(nk).finditer(text)]
            if pos:
                sp["ner:" + nk] = ("ner_" + lab.lower(), pos)
        return sp

    def occurrences(key, src, k):
        """character starts of a specific in any text (regex keys by pattern, NER keys by string)."""
        text = texts[(src, k)]
        if key.startswith("ner:"):
            return [m.start() for m in ner_pattern(key[4:]).finditer(text)]
        return regex_cache[(src, k)].get(key, (None, []))[1]

    regex_cache = {key: regex_specifics(t) for key, t in texts.items()}

    # --- reuse vs distance
    records = defaultdict(list)                   # system -> list of dicts
    excluded = defaultdict(int)
    for s in systems:
        for k in ids:
            ref_sp = specifics("ref", k)
            out_text = texts[(s, k)]
            ref_tok_prefix = {}
            for key, (typ, ref_pos) in ref_sp.items():
                ppos = occurrences(key, "prompt", k)
                if not ppos:
                    continue
                last = max(ppos)
                if last < vis_char[s][k]:
                    excluded[s] += 1              # only in the part of the prompt the model never saw
                    continue
                dist = tok_distance(k, last)
                first_ref = min(ref_pos)
                if first_ref not in ref_tok_prefix:
                    ref_tok_prefix[first_ref] = len(tok(refs[k]["text"][:first_ref], add_special_tokens=False)["input_ids"])
                hit = bool(occurrences(key, s, k))
                records[s].append(dict(id=k, key=key, type=typ, dist=dist, need_dist=dist + ref_tok_prefix[first_ref],
                                       hit=hit, vis_len=vis_len[k] if s == "model" else None))

    def curve(recs, field="dist", subset=None):
        idx = {k: i for i, k in enumerate(ids)}
        hits = np.zeros((len(ids), len(BUCKET_NAMES)))
        tots = np.zeros_like(hits)
        for r in recs:
            if subset and not subset(r):
                continue
            b = bucket_of(r[field])
            tots[idx[r["id"]], b] += 1
            hits[idx[r["id"]], b] += r["hit"]
        rate = hits.sum(0) / np.maximum(tots.sum(0), 1)
        lo, hi = cluster_bootstrap(hits, tots)
        return dict(n=tots.sum(0).astype(int).tolist(), hits=hits.sum(0).astype(int).tolist(),
                    rate=rate.tolist(), lo=lo.tolist(), hi=hi.tolist())

    results = {"curves": {}, "excluded_invisible": dict(excluded)}
    for s in systems:
        results["curves"][s] = {
            "all": curve(records[s]),
            "regex_only": curve(records[s], subset=lambda r: not r["type"].startswith("ner")),
            "ner_only": curve(records[s], subset=lambda r: r["type"].startswith("ner")),
            "need_dist": curve(records[s], field="need_dist"),
        }

    # --- decision rule
    def decide(c):
        mids = np.array([64, 192, 320, 448])
        y, w = np.array(c["rate"][:4]), np.array(c["n"][:4], dtype=float)
        slope, icpt = np.polyfit(mids, y, 1, w=np.sqrt(w))
        pred640 = slope * 640 + icpt
        r5, lo5, hi5 = c["rate"][4], c["lo"][4], c["hi"][4]
        lo4, hi4 = c["lo"][3], c["hi"][3]
        # the within-window decay is convex, so also fit log(rate) ~ distance (exponential trend)
        eslope, eicpt = np.polyfit(mids, np.log(np.maximum(y, 1e-3)), 1, w=np.sqrt(w))
        pred640_exp = float(np.exp(eslope * 640 + eicpt))
        below = r5 < pred640
        no_overlap = hi5 < lo4
        return dict(slope_per_100tok=slope * 100, intercept=icpt, extrapolated_512_768=pred640,
                    observed_512_768=r5, ci_512_768=[lo5, hi5], ci_384_512=[lo4, hi4],
                    below_trend=bool(below), ci_no_overlap=bool(no_overlap),
                    extrapolated_768_1024=slope * 896 + icpt, observed_768_1024=c["rate"][5],
                    extrapolated_512_768_exp=pred640_exp, below_trend_exp=bool(r5 < pred640_exp),
                    exp_intercept=float(eicpt), exp_slope=float(eslope),
                    conclusion="hybrid is hurting" if below and no_overlap else "no clear evidence",
                    conclusion_exp_fit="hybrid is hurting" if r5 < pred640_exp and no_overlap else "no clear evidence")
    results["decision"] = {s: decide(results["curves"][s]["all"]) for s in systems}
    results["decision_regex_only"] = decide(results["curves"]["model"]["regex_only"])
    results["label"], results["pred"] = args.label, os.path.abspath(args.pred)

    # --- supporting stats
    vl = np.array([vis_len[k] for k in ids])
    fl = np.array([n_tok[k] for k in ids])
    results["prompt_lengths"] = dict(
        full=dict(zip(["min", "p25", "median", "p75", "max", "mean"],
                      [int(fl.min()), *np.percentile(fl, [25, 50, 75]).tolist(), int(fl.max()), float(fl.mean())])),
        visible=dict(zip(["min", "p25", "median", "p75", "max", "mean"],
                         [int(vl.min()), *np.percentile(vl, [25, 50, 75]).tolist(), int(vl.max()), float(vl.mean())])),
        n_full_gt512=int((fl > 512).sum()), n_visible_gt512=int((vl > 512).sum()),
        n_visible_gt1024=int((vl > 1024).sum()), n_truncated_by_generate=int((vl < fl).sum()),
        hist_visible=np.histogram(vl, bins=[0, 256, 512, 768, 1024, 1216, 1217])[0].tolist())
    for s in systems:
        recs = records[s]
        results.setdefault("share_beyond_512", {})[s] = dict(
            n_reused_specifics=len(recs), n_beyond_512=sum(r["dist"] > WINDOW for r in recs),
            share=sum(r["dist"] > WINDOW for r in recs) / max(len(recs), 1),
            share_need_dist=sum(r["need_dist"] > WINDOW for r in recs) / max(len(recs), 1),
            by_type={t: dict(n=sum(r["type"] == t for r in recs),
                             hit_rate=float(np.mean([r["hit"] for r in recs if r["type"] == t] or [np.nan])))
                     for t in sorted({r["type"] for r in recs})})

    med = float(np.median(vl))
    SPLITS = [("<512", lambda k: vis_len[k] < 512), (">=512", lambda k: vis_len[k] >= 512),
              (f"<{med:.0f} (median)", lambda k: vis_len[k] < med), (f">={med:.0f} (median)", lambda k: vis_len[k] >= med)]

    # invented specifics: output specifics that occur in neither the (full) prompt nor the reference
    for s in systems:
        rows = []
        for k in ids:
            osp = specifics(s, k)
            inv = [key for key in osp if not occurrences(key, "prompt", k) and not occurrences(key, "ref", k)]
            rows.append((k, len(osp), len(inv), len([x for x in inv if not x.startswith("ner:")]),
                         len([x for x in osp if not x.startswith("ner:")])))
        res = {}
        for name, cond in SPLITS:
            sel = [r for r in rows if cond(r[0])]
            if not sel:
                res[name] = dict(n_prompts=0)
                continue
            tot, inv = np.array([r[1] for r in sel]), np.array([r[2] for r in sel])
            rtot, rinv = np.array([r[4] for r in sel]), np.array([r[3] for r in sel])
            rng = np.random.default_rng(0)
            bi = rng.integers(0, len(sel), size=(2000, len(sel)))
            boots = inv[bi].sum(1) / np.maximum(tot[bi].sum(1), 1)
            res[name] = dict(n_prompts=len(sel), n_output_specifics=int(tot.sum()), n_invented=int(inv.sum()),
                             invented_rate=float(inv.sum() / max(tot.sum(), 1)),
                             ci=np.percentile(boots, [2.5, 97.5]).tolist(),
                             invented_rate_regex_only=float(rinv.sum() / max(rtot.sum(), 1)),
                             invented_per_output=float(inv.mean()))
        results.setdefault("invented", {})[s] = res

    # BERTScore per 32-word window, split by prompt length
    if not args.no_bertscore:
        import bert_score
        scorer = bert_score.BERTScorer(lang="en", batch_size=128, device="cuda")
        for s in systems:
            res = {}
            wins = [(0, 32), (32, 64), (64, 96), (96, 128)]
            cand, ref, meta = [], [], []
            for k in ids:
                ow, rw = texts[(s, k)].split(), refs[k]["text"].split()
                for a, b in wins:
                    cand.append(" ".join(ow[a:b]) or "."); ref.append(" ".join(rw[a:b]) or ".")
                    meta.append((k, f"{a + 1}-{b}"))
            _, _, f = scorer.score(cand, ref)
            f = f.tolist()
            for grp, cond in SPLITS + [("all", lambda k: True)]:
                res[grp] = {}
                if not any(cond(k) for k in ids):
                    continue
                for a, b in wins:
                    w = f"{a + 1}-{b}"
                    vals = np.array([x for x, (k, ww) in zip(f, meta) if ww == w and cond(k)])
                    bi = np.random.default_rng(0).integers(0, len(vals), size=(2000, len(vals)))
                    res[grp][w] = dict(mean=float(vals.mean()), ci=np.percentile(vals[bi].mean(1), [2.5, 97.5]).tolist(),
                                       n=len(vals))
            results.setdefault("bertscore_windows", {})[s] = res

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    with open(os.path.join(args.out, "records.jsonl"), "w") as fh:
        for s in systems:
            for r in records[s]:
                fh.write(json.dumps(dict(system=s, **r)) + "\n")
    plot(results, args.out)
    print(json.dumps(results["decision"], indent=1))
    print("wrote", args.out)


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"model": "#2a78d6", "llama": "#eb6834"}
    LABEL = {"model": results.get("label", "model"), "llama": "Llama (all global)"}
    INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    x = np.arange(len(BUCKET_NAMES))
    mids = np.array([64, 192, 320, 448, 640, 896, 1152])

    def one(ax, curves, title, xlabel, trend=False):
        for i, s in enumerate(curves):
            c = curves[s]
            n = np.array(c["n"])
            keep = n > 0                                      # empty buckets are not plotted
            r, lo, hi = (np.array(c[f], dtype=float)[keep] for f in ("rate", "lo", "hi"))
            off = (i - (len(curves) - 1) / 2) * 0.12
            ax.errorbar(x[keep] + off, r, yerr=[r - lo, hi - r], color=COL[s], lw=2, marker="o", ms=7,
                        capsize=3, mec="white", mew=1.5, label=LABEL[s])
            for xi, ni in zip(x, n):
                ax.text(xi, -0.035, f"n={ni}", ha="center", fontsize=8, color=MUTED)
            if trend and s == "model":
                d = results["decision"]["model"]
                xs = np.linspace(0, 4, 50)
                ms = np.interp(xs, x[:5], mids[:5])
                lin = d["intercept"] + d["slope_per_100tok"] / 100 * ms
                ax.plot(xs[lin >= 0], lin[lin >= 0], ls="--", lw=1.2, color=MUTED,
                        label="linear fit on 0-512, extrapolated")
                ax.plot(xs, np.exp(d["exp_intercept"] + d["exp_slope"] * ms), ls=":", lw=1.6, color=INK,
                        label="exponential fit on 0-512, extrapolated")
        ax.axvline(3.5, color=INK, lw=1, ls=(0, (1, 3)))
        ax.text(3.55, 0.97, "window = 512", transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=INK)
        ax.set_xticks(x, BUCKET_NAMES)
        ax.set_xlabel(xlabel, color=MUTED, labelpad=14)
        ax.set_ylabel("reuse rate (specific appears in output)", color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_ylim(-0.05, None)
        ax.set_facecolor(SURF)

    curves = results["curves"]
    dist_label = "tokens from last prompt occurrence to prompt end"
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURF)
    one(ax, {s: curves[s]["all"] for s in curves},
        "Reuse of prompt specifics that the reference also uses (95% bootstrap CI)", dist_label, trend=True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reuse_vs_distance.png"), dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=SURF)
    for ax, key, title, xl in zip(axes, ["regex_only", "ner_only", "need_dist"],
                                  ["Regex specifics only", "NER entities only", "Distance at time of need"],
                                  [dist_label, dist_label,
                                   "prompt distance + tokens into the reference before first use"]):
        one(ax, {s: curves[s][key] for s in curves}, title, xl)
    if len(curves) > 1:
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reuse_vs_distance_breakdown.png"), dpi=150)


if __name__ == "__main__":
    main()
