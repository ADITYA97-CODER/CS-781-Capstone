"""Specifics in an SFT target, supported (in the prompt) or not, mapped to target tokens.

Uses the Phase 1 extractor (phase1/specifics.py):
  - regex specifics with normalized keys: "Sec. 302" / "Section 302" / "u/s 302" -> sec:302, "PW 4" / "PW-4" -> wit:PW-4,
    dates -> date:YYYY-MM-DD, citations, Acts + year, articles, rules, amounts;
  - spaCy NER names (PERSON/ORG/GPE/LOC/FAC, generic ones removed), normalized with ner_norm and looked up in the
    prompt case-insensitively (ner_pattern), exactly as the Phase 1 reuse / invented analysis does.
supported   = the specific's key occurs in the prompt
unsupported = it does not
For section/article/rule lists ("Sections 302, 307 and 34 IPC") each number is its own specific with its own span,
so a supported 302 is not masked because 307 is unsupported. Every other specific's span is its whole match.

label modes (flag mask_mode):
  none                    all target tokens trained, weight 1
  unsupported             tokens overlapping an unsupported span -> label -100
  unsupported+upweight    as above, and tokens overlapping a supported span (and no unsupported one) get
                          weight supported_weight
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from specifics import PATTERNS, ner_norm, ner_ok, ner_pattern, regex_specifics  # noqa: E402

MASK_MODES = ("none", "unsupported", "unsupported+upweight")
LIST_PREFIX = {"section": "sec", "article": "art", "rule": "rule"}
NUM = re.compile(r"\d+[A-Za-z]{0,2}")


def target_specifics(prompt, target, target_doc=None):
    """-> list of dicts(start, end, type, key, supported) for every specific occurrence in `target`
    (character offsets into `target`). target_doc: spaCy Doc of `target` (None = regex only)."""
    prompt_keys = set(regex_specifics(prompt))
    out = []
    for typ, pat, fn in PATTERNS:
        for m in pat.finditer(target):
            keys = fn(m)
            if not keys:
                continue
            if typ in LIST_PREFIX:
                base = m.start(1)
                for nm in NUM.finditer(m.group(1)):
                    key = f"{LIST_PREFIX[typ]}:{nm.group().upper()}"
                    out.append(dict(start=base + nm.start(), end=base + nm.end(), type=typ, key=key,
                                    supported=key in prompt_keys))
            else:
                a, b = m.span()
                b = a + len(target[a:b].rstrip())
                for key in keys:
                    out.append(dict(start=a, end=b, type=typ, key=key, supported=key in prompt_keys))
    if target_doc is not None:
        for e in target_doc.ents:
            if ner_ok(e):
                k = ner_norm(e.text)
                out.append(dict(start=e.start_char, end=e.end_char, type="ner_" + e.label_.lower(), key="ner:" + k,
                                supported=bool(ner_pattern(k).search(prompt))))
    return out


def token_flags(offsets, specifics):
    """offsets: (n_tok, 2) char offsets of the target tokens (relative to the target string).
    -> (unsupported, supported) bool arrays: token overlaps an unsupported / supported specific span.
    A token overlapping both counts as unsupported only."""
    offs = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    s, e = offs[:, 0], offs[:, 1]
    uns = np.zeros(len(offs), dtype=bool)
    sup = np.zeros(len(offs), dtype=bool)
    for sp in specifics:
        hit = (s < sp["end"]) & (e > sp["start"])
        if sp["supported"]:
            sup |= hit
        else:
            uns |= hit
    return uns, sup & ~uns


def target_labels(target_ids, unsupported, supported, mask_mode, supported_weight=2.0):
    """-> (labels int64 with -100 on masked tokens, weights float32) for the target tokens."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
    labels = np.asarray(target_ids, dtype=np.int64).copy()
    w = np.ones(len(labels), dtype=np.float32)
    if mask_mode != "none":
        labels[np.asarray(unsupported, dtype=bool)] = -100
    if mask_mode == "unsupported+upweight":
        w[np.asarray(supported, dtype=bool) & (labels != -100)] = supported_weight
    return labels, w
