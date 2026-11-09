"""Specifics of a prompt as token ids, for prompt-boost decoding (generate.py --prompt-boost).

Specifics are found with the Phase-1 extractor of specifics.py: regex patterns for
citations, sections/articles/rules, Acts, witness ids, dates and amounts, plus spaCy NER names
(PERSON/ORG/GPE/LOC/FAC, generic ones removed). Each specific is tokenized exactly as it appears in
the prompt: its tokens are the prompt tokens (legal_bpe_32k) overlapping its character span, so a
mid-sentence name keeps its leading-space token form.

Per prompt:
  seqs  token-id sequence of every distinct specific (deduplicated; capped at MAX_SEQ_LEN tokens)
  ids   union of those token ids minus common tokens: special ids, ids in the top 2% by document
        frequency in the training corpus, whitespace/punctuation-only tokens and stopwords

Usage (build and cache; generate.py also builds the cache on first use):
  python phase1/prompt_facts.py --prompts local --out outputs/phase1/fact_decoding/facts_local.json
"""

import argparse
import json
import os
import string
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from specifics import PATTERNS, ner_ok  # noqa: E402

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKEN_DIR = os.path.join(PROJECT_DIR, "data", "tokens")
DF_CACHE = os.path.join(PROJECT_DIR, "outputs", "phase1", "fact_decoding", "token_df_{tok}.npy")
SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
N_SPECIAL = 105
MAX_SEQ_LEN = 24


def token_doc_freq(tok_name, docs_per_source=20000, seed=0):
    """Fraction of training documents containing each token id (random docs from every source)."""
    path = DF_CACHE.format(tok=tok_name)
    if os.path.exists(path):
        return np.load(path)
    rng = np.random.default_rng(seed)
    df, n_docs = None, 0
    for s in SOURCES:
        toks = np.memmap(os.path.join(TOKEN_DIR, tok_name, f"{s}.train.bin"), dtype=np.uint16, mode="r")
        starts = np.load(os.path.join(TOKEN_DIR, tok_name, f"{s}.train.idx.npy"))
        ends = np.append(starts[1:], len(toks))
        for d in rng.choice(len(starts), size=min(docs_per_source, len(starts)), replace=False):
            u = np.unique(toks[starts[d]:ends[d]])
            if df is None:
                df = np.zeros(65536, dtype=np.int64)
            df[u] += 1
            n_docs += 1
    df = df / n_docs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, df)
    return df


def common_token_ids(tokenizer, tok_name, top_frac=0.02):
    """Ids never start-boosted: specials, top `top_frac` by document frequency, punctuation/whitespace
    tokens and stopwords."""
    from spacy.lang.en.stop_words import STOP_WORDS
    V = len(tokenizer)
    df = token_doc_freq(tok_name)[:V]
    common = set(range(N_SPECIAL)) | set(np.argsort(-df)[:int(round(top_frac * V))].tolist())
    for i in range(N_SPECIAL, V):
        s = tokenizer.decode([i]).strip().lower()
        if not s or all(c in string.punctuation for c in s) or s in STOP_WORDS:
            common.add(i)
    return common


def specific_spans(text, doc=None):
    """(start, end, type) character spans of every specific occurrence in `text`."""
    spans = set()
    for typ, pat, fn in PATTERNS:
        for m in pat.finditer(text):
            if fn(m):
                a, b = m.span()
                spans.add((a, a + len(text[a:b].rstrip()), typ))
    if doc is not None:
        spans |= {(e.start_char, e.end_char, "ner_" + e.label_.lower()) for e in doc.ents if ner_ok(e)}
    return sorted(spans)


def prompt_facts(text, tokenizer, common, doc=None):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    starts = np.array([o[0] for o in offs])
    ends = np.array([o[1] for o in offs])
    seqs, seen = [], set()
    for a, b, _ in specific_spans(text, doc):
        sel = np.nonzero((starts < b) & (ends > a))[0]
        seq = tuple(ids[j] for j in sel[:MAX_SEQ_LEN])
        if seq and seq not in seen:
            seen.add(seq)
            seqs.append(list(seq))
    fact_ids = sorted({t for s in seqs for t in s} - common)
    return {"seqs": seqs, "ids": fact_ids}


def build_facts(prompts, tokenizer, tok_name, use_ner=True):
    """prompts: {id: {"text": ...}} -> {id: {"seqs": [[...]], "ids": [...]}}"""
    common = common_token_ids(tokenizer, tok_name)
    keys = list(prompts)
    docs = [None] * len(keys)
    if use_ner:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        nlp.max_length = 2_000_000
        docs = list(nlp.pipe((prompts[k]["text"] for k in keys), batch_size=32, n_process=8))
    return {k: prompt_facts(prompts[k]["text"], tokenizer, common, d) for k, d in zip(keys, docs)}


def load_or_build(path, prompts, tokenizer, tok_name):
    if os.path.exists(path):
        facts = json.load(open(path))
        if set(prompts) <= set(facts):
            return facts
    facts = build_facts(prompts, tokenizer, tok_name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(facts, open(path, "w"))
    return facts


if __name__ == "__main__":
    from transformers import PreTrainedTokenizerFast
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="legal_bpe_32k")
    a = ap.parse_args()
    files = {"official": os.path.join(PROJECT_DIR, "data", "Phase_1_Pretraining", "Test", "test.json"),
             "local": os.path.join(PROJECT_DIR, "data", "corpus", "eval", "local_test.json")}
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(PROJECT_DIR, "artifacts", "tokenizer", a.tokenizer))
    pr = json.load(open(files.get(a.prompts, a.prompts)))
    f = load_or_build(a.out, pr, tok, a.tokenizer)
    n_seq = [len(v["seqs"]) for v in f.values()]
    n_ids = [len(v["ids"]) for v in f.values()]
    print(f"{len(f)} prompts | specifics per prompt: mean {np.mean(n_seq):.1f}, max {max(n_seq)} | "
          f"boosted ids per prompt: mean {np.mean(n_ids):.1f}, max {max(n_ids)}")
