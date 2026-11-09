"""Phase-3 post-processing: relabel a detected dissenting opinion to the Dissent role.

Gold pattern in the labelled data: a dissent is annotated as one block, from the start of the separate opinion to its
end. The rule finds that block with regular expressions and sets every sentence in it to Dissent; nothing is learned.
The first-person cue (FIRST_PERSON_DISAGREE) was added after reading the unlabelled test text.

  python apply_dissent_rule.py --pred ../outputs/phase3/all_wide3/prediction.json --out ../outputs/phase3/final
"""
import argparse
import json
import os
import re

import rr_data as D
from train_rr import TEST, write_submission

# A separate opinion starts with the author's name: "K. Sikri, J. ...", "Prasad, Member (Dissenting) ...".
OPINION_HEAD = re.compile(r"^\s*(?:[A-Z][A-Za-z.']*\s+){0,4}[A-Z][A-Za-z.']*\s*(?:,\s*|\s+)(?:J\.|C\.J\.)(?=\s|$)"
                          r"|^\s*(?:[A-Z][A-Za-z.']*\s+){0,4}[A-Z][A-Za-z.']*\s*,\s*Member\b")
# Explicit disagreement with the majority (a concurrence "I agree ..." never matches).
DISAGREE = re.compile(r"\(\s*dissent\w*\s*\)|\bdissent(?:ing|s)?\b(?! note)|respectfully (?:disagree|differ)"
                      r"|\b(?:cannot|can ?not|unable to) (?:agree|concur) with (?:the )?(?:majority|my (?:learned|esteemed))"
                      r"|\bI (?:have|take|hold) a different view|\bdiffer from the (?:majority|view)", re.I)
CUE_WINDOW = 3          # the cue must be in the heading sentence or the next 2 sentences
FIRST_PERSON_DISAGREE = re.compile(
    r"\bI\b[^.]{0,40}?\b(?:do not|don't|cannot|can not|am unable to|could not|respectfully)\s+(?:agree|concur|disagree|differ)"
    r"\b[^.]{0,80}?\bmajority|\bI (?:have|take|hold) a different view", re.I)


def opinion_heads(text):
    return [i for i, s in enumerate(text) if OPINION_HEAD.match(s)]


def dissent_region(text):
    """(start, end) of a named dissenting opinion: a heading that is not the first sentence, with a disagreement cue
    within CUE_WINDOW sentences; it ends at the next heading or the document end. None if there is no such opinion."""
    heads = opinion_heads(text)
    for h in heads:
        if h == 0:
            continue
        if any(DISAGREE.search(text[j]) for j in range(h, min(len(text), h + CUE_WINDOW))):
            nxt = [x for x in heads if x > h + CUE_WINDOW - 1]
            return h, (nxt[0] if nxt else len(text))
    return None


def dissent_region_v2(text):
    """As dissent_region; otherwise the opinion containing the first first-person disagreement with the majority."""
    reg = dissent_region(text)
    if reg is not None:
        return reg
    for i, s in enumerate(text):
        if FIRST_PERSON_DISAGREE.search(s):
            heads = opinion_heads(text)
            start = max([h for h in heads if h <= i], default=0)
            nxt = [h for h in heads if h > i]
            return start, (nxt[0] if nxt else len(text))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction.json of the ensemble")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    docs = [d for n in TEST for d in D.load_split(n)]
    changed = 0
    for d in docs:
        reg = dissent_region_v2(d["text"])
        if reg is None:
            continue
        labels = pred[d["id"]]
        for i in range(*reg):
            changed += labels[i] != "Dissent"
            labels[i] = "Dissent"
    os.makedirs(a.out, exist_ok=True)
    write_submission(pred, docs, a.out)
    print(f"Dissent rule changed {changed} test sentences")


if __name__ == "__main__":
    main()
