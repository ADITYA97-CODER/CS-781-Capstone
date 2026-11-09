"""Facts block for the ceiling test: prompt specifics written as plain text in front of the prompt.

    FACTS: <fact 1> ; <fact 2> ; ... END FACTS\n<prompt><continuation>

No new tokens: the block is ordinary text (legal_bpe_32k has a "\n" token), tokenized on its own and placed before
the prompt ids, which are unchanged. An empty block is "FACTS: END FACTS\n".

Specifics come from the PROMPT only, with the Phase 1 extractor (attention_reach.py): regex specifics (normalized
keys) and spaCy NER names (ner_norm keys, ner_ok filter). One fact per key, written as it appears in the prompt at its
LAST occurrence (section/article/rule lists: the list word as written + that number, e.g. "Sections 307"; whole-match
surfaces for everything else). Facts with the same surface are merged. Order = last position in the prompt (most
recent last). A fact is "used" if any of its keys appears in the continuation (regex key / NER pattern).
The block is capped at MAX_BLOCK_TOKENS: the oldest facts are dropped first.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagnostics"))
from attention_reach import PATTERNS, ner_norm, ner_ok, ner_pattern, regex_specifics  # noqa: E402

MAX_BLOCK_TOKENS = 150
MAX_PROMPT_TOKENS = 1536 - 320          # generate.py: max_seq_len - max_new_tokens
MAX_TOTAL_TOKENS = 1535                 # + <|endoftext|> separator = seq_len 1536
LIST_PREFIX = {"section": "sec", "article": "art", "rule": "rule"}
NUM = re.compile(r"\d+[A-Za-z]{0,2}")
MIX = (("oracle", 0.5), ("all", 0.3), ("empty", 0.2))
P_DROP, P_DISTRACT = 0.15, 0.2


def ner_keys(doc):
    """spaCy Doc (or object with .ents) -> sorted normalized NER keys (as Ctx.run_ner)."""
    return sorted({ner_norm(e.text) for e in doc.ents if ner_ok(e)})


def prompt_facts(prompt, nkeys):
    """-> list of facts dict(text, keys, pos) sorted by last position in the prompt (most recent last)."""
    by_key = {}                                   # key -> (pos, surface): keep the last occurrence
    for typ, pat, fn in PATTERNS:
        for m in pat.finditer(prompt):
            keys = fn(m)
            if not keys:
                continue
            if typ in LIST_PREFIX:
                word = prompt[m.start():m.start(1)].strip()
                base = m.start(1)
                for nm in NUM.finditer(m.group(1)):
                    key = f"{LIST_PREFIX[typ]}:{nm.group().upper()}"
                    pos = base + nm.start()
                    if key not in by_key or pos >= by_key[key][0]:
                        by_key[key] = (pos, f"{word} {nm.group()}")
            else:
                surf = " ".join(m.group().split()).strip(" ,;")
                for key in keys:
                    if key not in by_key or m.start() >= by_key[key][0]:
                        by_key[key] = (m.start(), surf)
    for k in nkeys:
        last = None
        for m in ner_pattern(k).finditer(prompt):
            last = m
        if last is not None:
            by_key["ner:" + k] = (last.start(), " ".join(last.group().split()))
    merged = {}
    for key, (pos, surf) in by_key.items():
        f = merged.setdefault(surf, dict(text=surf, keys=[], pos=pos))
        f["keys"].append(key)
        f["pos"] = max(f["pos"], pos)
    return sorted(merged.values(), key=lambda f: (f["pos"], f["text"]))


def key_in(key, text, text_regex_keys):
    if key.startswith("ner:"):
        return bool(ner_pattern(key[4:]).search(text))
    return key in text_regex_keys


def mark_used(facts, continuation):
    rk = set(regex_specifics(continuation))
    for f in facts:
        f["used"] = any(key_in(k, continuation, rk) for k in f["keys"])
    return facts


def block_text(facts):
    return "FACTS: " + " ; ".join(f["text"] for f in facts) + " END FACTS\n" if facts else "FACTS: END FACTS\n"


def block_ids(tok, facts, cap=MAX_BLOCK_TOKENS):
    """-> (ids, facts kept). Drops the oldest facts until the block fits in `cap` tokens."""
    facts = list(facts)
    while True:
        ids = tok(block_text(facts), add_special_tokens=False)["input_ids"]
        if len(ids) <= cap or not facts:
            return ids, facts
        facts = facts[1:]


def truncate_prompt(prompt_ids, n_block, n_target):
    """Left-truncate the prompt so that block + prompt <= MAX_PROMPT_TOKENS and block + prompt + target
    <= MAX_TOTAL_TOKENS. -> (kept prompt ids, number of tokens dropped). The prompt's end is always kept."""
    keep = max(0, min(len(prompt_ids), MAX_PROMPT_TOKENS - n_block, MAX_TOTAL_TOKENS - n_block - n_target))
    return prompt_ids[len(prompt_ids) - keep:], len(prompt_ids) - keep


def choose_facts(rng, kind, facts, distractor_pool, noise=True):
    """kind oracle / all / empty -> facts list for the block. Noise (oracle and all only): drop each fact with
    P_DROP; with P_DISTRACT insert one fact from distractor_pool (other documents, pre-filtered) at a random place."""
    if kind == "empty":
        return []
    out = [f for f in facts if f["used"]] if kind == "oracle" else list(facts)
    if not noise:
        return out
    out = [f for f in out if rng.random() >= P_DROP]
    if distractor_pool and rng.random() < P_DISTRACT:
        d = dict(distractor_pool[int(rng.integers(len(distractor_pool)))], distractor=True, used=False)
        out.insert(int(rng.integers(len(out) + 1)), d)
    return out


def distractor_ok(fact, prompt, prompt_rk, target, target_rk):
    """A fact from another document is usable only if none of its keys occurs in this prompt or this continuation."""
    return not any(key_in(k, prompt, prompt_rk) or key_in(k, target, target_rk) for k in fact["keys"])
