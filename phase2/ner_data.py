"""Phase 2 (L-NER) data: documents -> token windows with BIO tags, and token tags -> character spans.

  * Tokens come from the Phase-1 tokenizer (legal_bpe_32k) with character offsets. Byte-level BPE
    puts the space before a word inside the token, so every token's start is moved past its
    leading whitespace ("core" offsets).
  * A token is tagged with an entity if its core overlaps the entity: B-<label> for the first
    such token, I-<label> for the rest. Entities never overlap in the training data.
  * Windows: [header tokens] <|endoftext|> [content tokens], padded to one fixed length. The
    header is the first `header_len` tokens of the document (parties, counsel, judges), given to
    every window as context and never scored. Content windows overlap; at prediction time every
    token keeps the scores of the window where it sits furthest from a content edge.
  * Decoding: constrained Viterbi (I-X only after B-X or I-X), then tokens -> characters with
    trailing-whitespace stripping and a few suffix trims learned from the training data
    (`trim_end`, `learn_trims`), e.g. "Railway Board|s", "...Ltd.|,", "...1728]|.".
"""

import json
import os
import re

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_2_LNER")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizer", "legal_bpe_32k", "tokenizer.json")
LABELS = [l.strip() for l in open(os.path.join(DATA, "Train", "labels.txt")) if l.strip()]
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
EOS_ID = 0


def load_split(name):
    """'fold1' | 'fold2' | 'fold3' -> (texts {id: text}, gold {id: [[s, e, label, text], ...]} or None)."""
    sub = "Test" if name == "fold3" else "Train"
    texts = json.load(open(os.path.join(DATA, sub, f"{name}-text.json")))
    gold_path = os.path.join(DATA, sub, f"{name}-gold.json")
    gold = json.load(open(gold_path)) if os.path.exists(gold_path) else None
    return texts, gold


def tokenizer():
    return Tokenizer.from_file(TOKENIZER)


def encode_doc(tok, text):
    """-> ids (n,), core offsets (n, 2): each token's character span without leading whitespace."""
    enc = tok.encode(text, add_special_tokens=False)
    offs = []
    for a, b in enc.offsets:
        while a < b and text[a].isspace():
            a += 1
        offs.append((a, b))
    return np.array(enc.ids, dtype=np.int64), np.array(offs, dtype=np.int64)


def bio_tags(offs, entities):
    """Token tag ids for one document; tokens that are only whitespace get O."""
    tags = np.zeros(len(offs), dtype=np.int64)
    starts = offs[:, 0]
    for s, e, label, *_ in sorted(entities):
        lo = np.searchsorted(offs[:, 1], s, side="right")       # first token ending after s
        first = True
        for t in range(lo, len(offs)):
            a, b = offs[t]
            if a >= e:
                break
            if a >= b or b <= s or tags[t] != 0:
                continue
            tags[t] = TAG2ID[("B-" if first else "I-") + label]
            first = False
    del starts
    return tags


def gold_token_spans(offs, entities):
    """Entities -> [(first_token, last_token, label)]: first/last tokens whose core overlaps the entity.
    Used as span-head targets; decoding these spans gives the same characters as the BIO round trip."""
    out = []
    for s, e, label, *_ in sorted(entities):
        lo = np.searchsorted(offs[:, 1], s, side="right")
        toks = [t for t in range(lo, len(offs)) if offs[t, 0] < e and offs[t, 0] < offs[t, 1]
                and offs[t, 1] > s] if lo < len(offs) else []
        toks = [t for t in toks if offs[t, 0] < e]
        if toks:
            out.append((toks[0], toks[-1], label))
    return out


# ----------------------------------------------------------------------------- windows

def make_windows(n_tokens, content_len, stride):
    """Content windows [lo, hi) covering 0..n_tokens, consecutive starts `stride` apart."""
    if n_tokens <= content_len:
        return [(0, n_tokens)]
    out, lo = [], 0
    while True:
        hi = min(lo + content_len, n_tokens)
        out.append((hi - content_len, hi) if hi - lo < content_len else (lo, hi))
        if hi == n_tokens:
            return out
        lo += stride


class DocWindows:
    """All windows of a set of documents as fixed-length tensors.
    Row fields: ids, labels (-100 outside content), valid (attention keys), and for prediction the
    (doc, lo, hi) of its content and where its content starts inside the row."""

    def __init__(self, docs, header_len, content_len, stride):
        """docs: list of (doc_id, ids, offs, tags or None[, gold token spans or None]).
        With gold token spans, `spans[row]` lists the spans inside the row's content, in content coordinates."""
        self.seq_len = header_len + 1 + content_len
        self.header_len = header_len
        ids, labels, valid, meta, row_spans = [], [], [], [], []
        for di, d in enumerate(docs):
            d_ids, d_tags = d[1], d[3]
            d_spans = d[4] if len(d) > 4 else None
            head = d_ids[:header_len]
            # documents shorter than the header (one test doc is empty) get a header padded with masked
            # positions, so content always starts at header_len + 1 (the span head relies on it)
            head_ok = np.concatenate([np.zeros(header_len - len(head), dtype=bool), np.ones(len(head), dtype=bool)])
            head = np.concatenate([np.full(header_len - len(head), EOS_ID, dtype=np.int64), head])
            for lo, hi in make_windows(len(d_ids), content_len, stride):
                row = np.concatenate([head, [EOS_ID], d_ids[lo:hi]])
                lab = np.full(len(row), -100, dtype=np.int64)
                c0 = header_len + 1
                if d_tags is not None:
                    lab[c0:] = d_tags[lo:hi]
                pad = self.seq_len - len(row)
                ids.append(np.pad(row, (0, pad)))
                labels.append(np.pad(lab, (0, pad), constant_values=-100))
                valid.append(np.pad(np.concatenate([head_ok, np.ones(len(row) - header_len, dtype=bool)]), (0, pad)))
                meta.append((di, lo, hi, c0))
                row_spans.append([(i - lo, j - lo, l) for i, j, l in d_spans if lo <= i and j < hi]
                                 if d_spans is not None else None)
        self.spans = row_spans
        self.ids = torch.from_numpy(np.stack(ids))
        self.labels = torch.from_numpy(np.stack(labels))
        self.valid = torch.from_numpy(np.stack(valid))
        self.meta = meta

    def __len__(self):
        return len(self.meta)


# ----------------------------------------------------------------------------- FLERT windows

# a sentence ends at . ? ! (plus closing quotes/brackets) followed by whitespace and an upper-case
# letter, digit or opening quote/bracket; the texts have no newlines
_SENT_END = re.compile(r"[.?!][\"'”’)\]]*\s+(?=[\"“'(\[]?[A-Z0-9])")
_ABBREV = {"no", "nos", "rs", "v", "vs", "mr", "mrs", "ms", "dr", "sh", "smt", "shri", "ltd", "pvt", "co", "corpn",
           "corp", "inc", "art", "arts", "sec", "secs", "s", "ss", "cl", "para", "paras", "p", "pp", "vol", "ed",
           "etc", "viz", "i.e", "e.g", "hon'ble", "honble", "st", "govt", "dept", "addl", "asstt", "jt", "ors",
           "anr", "u/s", "cr", "crl", "civ", "spl", "w.p", "s.l.p", "c.a", "i.p.c", "cr.p.c", "a.i.r", "j", "jj", "c.j"}


def sentence_bounds(text, offs):
    """Token indices where sentences start (always 0 and n), from the character splitter above.
    Abbreviations and initials (one or two characters before the full stop) are not boundaries."""
    starts = [0]
    for m in _SENT_END.finditer(text):
        end = m.start() + 1                                  # just after the punctuation mark
        w = text[:end].rsplit(None, 1)[-1].lstrip("(\"“'[").rstrip(".?!").lower()
        if text[m.start()] == "." and (w in _ABBREV or len(w) <= 2):
            continue
        t = int(np.searchsorted(offs[:, 0], m.end(), side="left"))
        if starts[-1] < t < len(offs):
            starts.append(t)
    return starts + [len(offs)]


def flert_units(bounds, unit_len, pack=True):
    """Sentences [lo, hi) -> units of at most unit_len tokens: consecutive sentences packed greedily
    (pack=True) or one sentence per unit; a sentence longer than unit_len is cut into pieces."""
    out, lo = [], 0
    n = bounds[-1]
    while lo < n:
        nxt = [b for b in bounds if lo < b <= lo + unit_len]
        if not nxt:
            hi = min(lo + unit_len, n)
        else:
            hi = nxt[-1] if pack else nxt[0]
        out.append((lo, hi))
        lo = hi
    return out


class FlertWindows:
    """FLERT (Schweter & Akbik, 2020) document-level features: every row is one unit (a sentence or
    packed sentences) with `ctx` tokens of its own document on each side, preceded by the same
    document header as DocWindows: [header] <|endoftext|> [left ctx] [unit] [right ctx].
    Only unit tokens are labelled and predicted; each document token is in exactly one unit, so
    `merge_window_scores` just scatters them back. Same fields as DocWindows, plus `ctx` (B, T):
    the context positions (for context dropout in training)."""

    def __init__(self, docs, bounds, header_len, ctx, unit_len, pack=True):
        """docs: list of (doc_id, ids, offs, tags or None, ...); bounds: per doc, `sentence_bounds`."""
        self.seq_len = header_len + 1 + 2 * ctx + unit_len
        self.header_len = header_len
        self.spans = None
        ids, labels, valid, ctxm, meta = [], [], [], [], []
        for di, d in enumerate(docs):
            d_ids, d_tags = d[1], d[3]
            n = len(d_ids)
            head = d_ids[:header_len]
            head_ok = np.concatenate([np.zeros(header_len - len(head), dtype=bool), np.ones(len(head), dtype=bool)])
            head = np.concatenate([np.full(header_len - len(head), EOS_ID, dtype=np.int64), head])
            units = flert_units(bounds[di], unit_len, pack) if n else [(0, 0)]
            for lo, hi in units:
                a, b = max(0, lo - ctx), min(n, hi + ctx)
                row = np.concatenate([head, [EOS_ID], d_ids[a:b]])
                c0 = header_len + 1 + (lo - a)
                lab = np.full(len(row), -100, dtype=np.int64)
                if d_tags is not None:
                    lab[c0:c0 + hi - lo] = d_tags[lo:hi]
                cm = np.zeros(len(row), dtype=bool)
                cm[header_len + 1:] = True
                cm[c0:c0 + hi - lo] = False
                pad = self.seq_len - len(row)
                ids.append(np.pad(row, (0, pad)))
                labels.append(np.pad(lab, (0, pad), constant_values=-100))
                valid.append(np.pad(np.concatenate([head_ok, np.ones(len(row) - header_len, dtype=bool)]), (0, pad)))
                ctxm.append(np.pad(cm, (0, pad)))
                meta.append((di, lo, hi, c0))
        self.ids = torch.from_numpy(np.stack(ids))
        self.labels = torch.from_numpy(np.stack(labels))
        self.valid = torch.from_numpy(np.stack(valid))
        self.ctx = torch.from_numpy(np.stack(ctxm))
        self.meta = meta

    def __len__(self):
        return len(self.meta)


def merge_window_scores(n_tokens_per_doc, meta, row_scores):
    """row_scores: list of (seq_len, n_tags) log-prob arrays, one per window row.
    -> per document (n_tokens, n_tags): each token takes the window where its distance to the
    nearest content edge (a document edge counts as infinitely far) is largest."""
    out = [None] * len(n_tokens_per_doc)
    best = [None] * len(n_tokens_per_doc)
    for (di, lo, hi, c0), sc in zip(meta, row_scores):
        n = n_tokens_per_doc[di]
        if out[di] is None:
            out[di] = np.zeros((n, sc.shape[-1]), dtype=np.float32)
            best[di] = np.full(n, -1, dtype=np.int64)
        pos = np.arange(lo, hi)
        left = np.where(lo == 0, 10**9, pos - lo)
        right = np.where(hi == n, 10**9, hi - 1 - pos)
        dist = np.minimum(left, right)
        take = dist > best[di][lo:hi]
        out[di][lo:hi][take] = sc[c0:c0 + (hi - lo)][take]
        best[di][lo:hi][take] = dist[take]
    return out


# ----------------------------------------------------------------------------- decoding

def transition_mask():
    """allowed[i, j]: tag j may follow tag i. I-X only after B-X or I-X."""
    n = len(TAGS)
    allowed = np.ones((n, n), dtype=bool)
    for j, t in enumerate(TAGS):
        if t.startswith("I-"):
            allowed[:, j] = False
            allowed[TAG2ID["B-" + t[2:]], j] = True
            allowed[j, j] = True
    start_ok = np.array([not t.startswith("I-") for t in TAGS])
    return allowed, start_ok


_ALLOWED, _START_OK = transition_mask()
_CRF = None          # learned CRF scores (trans (K, K), start (K,), end (K,)) used by `viterbi`; see set_crf


def set_crf(params):
    """params: None (plain constrained Viterbi) or (trans, start, end) arrays of a trained CRF layer."""
    global _CRF
    _CRF = None if params is None else tuple(np.asarray(x, dtype=np.float32) for x in params)


def viterbi(logp):
    """logp: (n, n_tags) log-probabilities -> best tag sequence under the BIO constraints."""
    n, k = logp.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    trans = np.where(_ALLOWED, 0.0, -1e9).astype(np.float32)
    score = logp[0] + np.where(_START_OK, 0.0, -1e9)
    if _CRF is not None:
        trans = trans + _CRF[0]
        score = score + _CRF[1]
    back = np.zeros((n, k), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + trans                       # (from, to)
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    if _CRF is not None:
        score = score + _CRF[2]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def tags_to_token_spans(tags):
    """-> [(first_token, last_token, label)] from a BIO sequence (an I-X without an open X span starts one)."""
    spans, cur = [], None
    for t, tid in enumerate(tags):
        tag = TAGS[tid]
        if tag.startswith("B-") or tag == "O" or cur is None or tag[2:] != cur[2]:
            if cur:
                spans.append(tuple(cur))
            cur = [t, t, tag[2:]] if tag != "O" else None
        else:
            cur[1] = t
    if cur:
        spans.append(tuple(cur))
    return spans


_PAIRS = {")": "(", "]": "[", "}": "{"}


def trim_end(text, s, e, label):
    """Strip what BPE glued onto the entity's last token: whitespace, separators and opening
    brackets/quotes (an entity never ends with them), unbalanced closing brackets, "." after a
    closing bracket, and word endings the annotation usually leaves out (`learn_trims`)."""
    changed = True
    while changed and e - s > 1:
        changed = False
        c = text[e - 1]
        if c.isspace() or c in ',;:([{"“”\'':
            e -= 1
            changed = True
        elif c in _PAIRS and text[s:e].count(_PAIRS[c]) < text[s:e].count(c):
            e -= 1
            changed = True
        elif c == "." and text[e - 2] in ")]}":
            e -= 1
            changed = True
    stem = _TRIM.get((label, text[s:e]))
    if stem is not None:
        e = s + len(stem)
    return s, e


# (label, glued string) -> annotated stem, for strings the training annotation more often cuts
# inside their last word ("Railway Board" in "Railway Boards") than keeps whole. See `learn_trims`.
_TRIM = {}


def learn_trims(texts, gold):
    """Count, per (label, string up to the end of its last word), how often the annotation stops
    inside that word vs keeps it whole; keep the trims that win."""
    cut, whole, stems = {}, {}, {}
    for k, ents in gold.items():
        text = texts[k]
        for s, e, label, *_ in ents:
            whole[(label, text[s:e])] = whole.get((label, text[s:e]), 0) + 1
            if e < len(text) and text[e].isalnum() and text[e - 1].isalnum():
                key = (label, text[s:e] + re.match(r"\w+", text[e:]).group(0))
                cut[key] = cut.get(key, 0) + 1
                stems[key] = text[s:e]
    _TRIM.clear()
    _TRIM.update({key: stems[key] for key, n in cut.items() if n > whole.get(key, 0)})


def snap_words(text, s, e):
    """Extend a span that starts or ends inside a word to the whole word: the model sometimes stops
    between sub-word pieces ("M.SATH|YANARAYANAN"). Known cut words are re-trimmed by `trim_end`.
    CV (7 Oct): +0.012..0.015 macro-F1 together with dropping 1-character spans."""
    def joined(a, b):            # same word: letter-letter or digit-digit (not footnotes like "Sethi2")
        return (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())
    while s > 0 and joined(text[s - 1], text[s]):
        s -= 1
    while e < len(text) and joined(text[e - 1], text[e]):
        e += 1
    return s, e


def propagate_doc(text, ents, min_len=6, min_count=2):
    """Gold labels are consistent within a document (every exact repeat of an annotated string is
    annotated with the same label). Add the unpredicted whole-word repeats of strings the model
    predicted at least `min_count` times, always with the same label, that overlap no prediction.
    The filter matters: propagating every predicted string copies fragments ("Special" as COURT).
    CV (7 Oct): 0.5161/0.5076 -> 0.5228/0.5234; majority-vote relabelling hurt (-0.004) and is not used."""
    counts = {}
    for s, e, l in ents:
        counts.setdefault(text[s:e], {}).setdefault(l, 0)
        counts[text[s:e]][l] += 1
    taken = sorted((s, e) for s, e, _ in ents)
    added = []
    for x, labs in counts.items():
        if len(x) < min_len or len(labs) != 1 or sum(labs.values()) < min_count:
            continue
        label = next(iter(labs))
        for m in re.finditer(re.escape(x), text):
            a, b = m.span()
            if (a > 0 and text[a - 1].isalnum() and text[a].isalnum()) or \
               (b < len(text) and text[b - 1].isalnum() and text[b].isalnum()):
                continue
            if any(s < b and e > a for s, e in taken):
                continue
            taken.append((a, b))
            added.append([a, b, label])
    return ents + added


def apply_bias(logp, bias):
    """Add a per-label score bias to the B-/I- columns of token log-probs (negative: predict that label less).
    bias: {label: value}; tuned on CV dev log-probs by tune_bias.py."""
    if not bias:
        return logp
    logp = logp.copy()
    for l, v in bias.items():
        logp[:, [TAG2ID["B-" + l], TAG2ID["I-" + l]]] += v
    return logp


def decode_doc(text, offs, logp, propagate=True):
    """Token log-probs of one document -> [[start, end, label], ...] character spans."""
    out, seen = [], set()
    for t0, t1, label in tags_to_token_spans(viterbi(logp)):
        s, e = snap_words(text, int(offs[t0, 0]), int(offs[t1, 1]))
        s, e = trim_end(text, s, e, label)
        if e - s >= 2 and (s, e, label) not in seen:       # 1-char spans are fragments
            seen.add((s, e, label))
            out.append([s, e, label])
    return propagate_doc(text, out) if propagate else out


# ----------------------------------------------------------------------------- scoring

def strict_scores(gold, pred):
    """Same computation as official_eval/Evaluation Script/evaluate_phase2.py.
    -> (macro_f1, macro_p, macro_r, {label: (p, r, f1, tp, fp, fn)})."""
    stats = {l: [0, 0, 0] for l in LABELS}
    for did, g_ents in gold.items():
        g = {tuple(x[:3]) for x in g_ents}
        p = {tuple(x[:3]) for x in pred.get(did, [])}
        for l in LABELS:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            stats[l][0] += len(gs & ps)
            stats[l][1] += len(ps - gs)
            stats[l][2] += len(gs - ps)
    per = {}
    for l, (tp, fp, fn) in stats.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = (pr, rc, f, tp, fp, fn)
    n = len(LABELS)
    return (sum(v[2] for v in per.values()) / n, sum(v[0] for v in per.values()) / n,
            sum(v[1] for v in per.values()) / n, per)


def oracle_logp(tags, n_tags=len(TAGS)):
    lp = np.full((len(tags), n_tags), -20.0, dtype=np.float32)
    lp[np.arange(len(tags)), tags] = 0.0
    return lp


if __name__ == "__main__":
    # Round-trip test: gold spans -> BIO tags -> decoded spans, on both training folds.
    tok = tokenizer()
    for fold in ("fold1", "fold2"):
        texts, gold = load_split(fold)
        learn_trims(*load_split("fold2" if fold == "fold1" else "fold1"))   # other fold only (no leak)
        pred, conflicts = {}, 0
        for k, text in texts.items():
            ids, offs = encode_doc(tok, text)
            tags = bio_tags(offs, gold[k])
            pred[k] = decode_doc(text, offs, oracle_logp(tags))
        f1, p, r, per = strict_scores(gold, pred)
        print(f"{fold}: oracle round-trip macro-F1 {f1:.4f}  P {p:.4f}  R {r:.4f}")
        for l, v in per.items():
            if v[2] < 0.999:
                print(f"   {l:10s} P {v[0]:.4f} R {v[1]:.4f} F1 {v[2]:.4f}  fp {v[4]} fn {v[5]}")
