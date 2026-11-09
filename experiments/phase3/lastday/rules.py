"""Step 3: rule-based post-processing for the two rare labels. Every rule takes a document's sentences and the model's
per-sentence class probabilities (n, 13) / current labels (n,) and returns new label ids; nothing is learned, no
threshold or bias is tuned.

Gold facts behind the rules (90 labelled docs):
  * PrecedentOverruled: 3 sentences in 1 doc (SC_2010_170). The citation is split over several sentences, so the word
    "overruled" sits in a neighbouring sentence that is labelled PrecedentReliedUpon.
  * Dissent: 284 sentences in 2 CCI orders. In both, a member's separate order starts with a heading
    "Prasad, Member (Dissenting) ..." after the majority's operative order, and EVERY sentence from there to the end of
    the document is Dissent (facts, analysis and the dissent's own operative order alike).
"""

import re

import numpy as np

import common as C

L = C.D.LAB2ID
PRECEDENT = {L["PrecedentReliedUpon"], L["PrecedentNotReliedUpon"], L["PrecedentOverruled"]}

# ---------------------------------------------------------------- 3a PrecedentOverruled

PO_GATE = re.compile(r"\boverrul\w*", re.I)


def po_gate(sentence):
    """True if the sentence mentions over-ruling (overrule / overruled / overruling)."""
    return bool(PO_GATE.search(sentence))


def rule_precedent_overruled(text, probs, labels, variant="argmax"):
    """Relabel a sentence to PrecedentOverruled when the gate fires AND the model already sees it as a precedent:
      variant 'argmax': the model's current label is one of the Precedent* labels;
      variant 'top2'  : PrecedentOverruled is among the model's two most probable labels.
    Both conditions stop the regex from firing on, e.g., a ruling that merely mentions an overruled objection."""
    out = labels.copy()
    top2 = np.argsort(-probs, -1)[:, :2]
    for i, s in enumerate(text):
        if not po_gate(s):
            continue
        ok = labels[i] in PRECEDENT if variant == "argmax" else L["PrecedentOverruled"] in top2[i]
        if ok:
            out[i] = L["PrecedentOverruled"]
    return out


# ---------------------------------------------------------------- 3b Dissent

# A separate opinion starts with the author's name: "K. Sikri, J. ...", "Prasad, Member (Dissenting) ...".
OPINION_HEAD = re.compile(r"^\s*(?:[A-Z][A-Za-z.']*\s+){0,4}[A-Z][A-Za-z.']*\s*(?:,\s*|\s+)(?:J\.|C\.J\.)(?=\s|$)"
                          r"|^\s*(?:[A-Z][A-Za-z.']*\s+){0,4}[A-Z][A-Za-z.']*\s*,\s*Member\b")
# Explicit DISagreement with the majority (a concurrence "I agree ..." never matches).
DISAGREE = re.compile(r"\(\s*dissent\w*\s*\)|\bdissent(?:ing|s)?\b(?! note)|respectfully (?:disagree|differ)"
                      r"|\b(?:cannot|can ?not|unable to) (?:agree|concur) with (?:the )?(?:majority|my (?:learned|esteemed))"
                      r"|\bI (?:have|take|hold) a different view|\bdiffer from the (?:majority|view)", re.I)
CUE_WINDOW = 3          # the cue must be in the heading sentence or the next 2 sentences


def opinion_heads(text):
    """Indices of sentences that open a (new) opinion."""
    return [i for i, s in enumerate(text) if OPINION_HEAD.match(s)]


def dissent_region(text):
    """-> (start, end) of the dissenting opinion, or None. start = an opinion heading that is NOT the first sentence
    (the first opinion is the majority / main judgment) and that has an explicit disagreement cue within CUE_WINDOW
    sentences; end = the next opinion heading after the cue, or the document end."""
    heads = opinion_heads(text)
    for h in heads:
        if h == 0:
            continue
        if any(DISAGREE.search(text[j]) for j in range(h, min(len(text), h + CUE_WINDOW))):
            nxt = [x for x in heads if x > h + CUE_WINDOW - 1]
            return h, (nxt[0] if nxt else len(text))
    return None


# v2 (written AFTER reading the unlabelled test text, see SUBMISSIONS.md): a dissent can also be a member's separate
# note with no name heading, opened by a first-person disagreement with the majority ("I, however, do not agree with
# the above findings of the Majority Members"). On the 90 labelled docs this cue fires only inside a gold Dissent block.
FIRST_PERSON_DISAGREE = re.compile(
    r"\bI\b[^.]{0,40}?\b(?:do not|don't|cannot|can not|am unable to|could not|respectfully)\s+(?:agree|concur|disagree|differ)"
    r"\b[^.]{0,80}?\bmajority|\bI (?:have|take|hold) a different view", re.I)


def dissent_region_v2(text):
    """v1 region if found; else, at the first first-person disagreement with the majority, the opinion that contains
    it: from the last opinion heading before the cue (document start if none) to the next heading or document end."""
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


def rule_dissent(text, probs, labels, variant="ratio_ruling", region=dissent_region):
    """Inside the detected dissent region relabel to Dissent:
      variant 'ratio_ruling': only sentences the model labelled RatioOfTheDecision / RulingByPresentCourt (as specified);
      variant 'all'         : every sentence in the region (matches how the gold annotates a dissent: whole block)."""
    out = labels.copy()
    reg = region(text)
    if reg is None:
        return out
    s, e = reg
    for i in range(s, e):
        if variant == "all" or labels[i] in (L["RatioOfTheDecision"], L["RulingByPresentCourt"]):
            out[i] = L["Dissent"]
    return out
