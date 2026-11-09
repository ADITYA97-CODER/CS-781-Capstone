"""LLM (Qwen2.5-7B-Instruct) labels for the specifics in SFT targets: schema, instruction, validation, span mapping,
held-out guard and the resumable JSONL store. Pure Python (+numpy): imported by the vLLM labeling driver
(label_specifics_llm.py, env cs781-vllm) and by the mask builder / reports / tests (env cs781).

Qwen is used ONLY to build training data: the guard (HeldoutGuard) refuses any pair that is not an SFT pair or that
shares a 30-word passage with one of the 990 held-out prompts or the 1,000 official test prompts.

Validation of one model answer (validate_items), each failure counted:
  - unparsable JSON / wrong item shape                         -> whole answer or item dropped; an answer cut off at
                                                                  max_tokens (a loop) keeps its complete items ("salvaged")
  - text empty, or not found verbatim in TARGET                -> dropped ("not_verbatim"); a text found only inside a
                                                                  longer word (e.g. "Ram" in "Ramesh") -> "no_word_match"
  - text is a generic legal phrase (STOPLIST / ROLE_RE)        -> dropped ("stoplist")
  - label supported but evidence missing / not verbatim in the PROMPT shown to the model
                                                               -> relabeled unsupported ("evidence_fail")
  - the same text listed twice                                 -> merged, label priority supported > inferable >
                                                                  unsupported ("duplicate", "label_conflict")
Every kept item is mapped to ALL its occurrences in TARGET (char spans, word-boundary aware).
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagnostics"))
from attention_reach import GENERIC, ner_norm  # noqa: E402

TYPES = ["name", "place", "court", "citation", "case_number", "section", "act", "witness", "date", "amount",
         "exhibit", "judge", "other"]
LABELS = ["supported", "inferable", "unsupported"]
SCHEMA = {
    "type": "object",
    "properties": {"specifics": {"type": "array", "items": {
        "type": "object",
        # evidence BEFORE label: the model first looks for the specific in PROMPT, then decides the label
        "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": TYPES},
                       "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                       "label": {"type": "string", "enum": LABELS}},
        "required": ["text", "type", "evidence", "label"], "additionalProperties": False}}},
    "required": ["specifics"], "additionalProperties": False}

MAX_PROMPT_QWEN_TOKENS = 6000

SYSTEM = """You are annotating Indian legal text. Each request gives PROMPT (the earlier part of a judgment, order or statute) and TARGET (the text that immediately follows it). Answer with JSON only.

List every specific in TARGET: person/party names, places, courts, case citations, case numbers, statute sections/articles/rules, Acts, witness ids (PW/DW), dates, amounts, exhibit numbers, judge names.

For each specific give:
- "text": the specific exactly as it appears in TARGET, copied character for character (same spelling, case, spacing and punctuation). Use the shortest span that contains the specific.
- "type": one of name, place, court, citation, case_number, section, act, witness, date, amount, exhibit, judge, other.
- "evidence": search PROMPT for the same specific (possibly written differently). If PROMPT states it, copy the PROMPT text that states it, character for character from PROMPT, never from TARGET (a short span of at most about 15 words that contains the specific). If PROMPT does not state it, null.
- "label":
  - "supported": stated in PROMPT, possibly in a different form (e.g. "302/34" vs "Section 302 read with Section 34 IPC", "Ramlal" vs "Ram Lal", "PW-4" vs "P.W.4"). Its evidence is the PROMPT text.
  - "inferable": not stated in PROMPT, but directly derivable from PROMPT (e.g. a party referred to by role, a court implied by the case). Evidence null.
  - "unsupported": new information; PROMPT gives no way to know it. Evidence null.
Most TARGETs contain both supported and unsupported specifics: a specific is only supported if it really occurs in PROMPT.

Rules:
- Go through TARGET from beginning to end and list every specific in it. Only list text that occurs in TARGET; text that occurs only in PROMPT is never listed.
- Do not list generic legal words or roles without a name (e.g. "the Court", "the appellant", "learned counsel", "the Act", "this Tribunal").
- List each distinct specific once, even if it occurs several times in TARGET. Never repeat an item.
- If a list mixes supported and unsupported items (e.g. "Sections 302, 307 and 34"), list the items separately, each with only the part written in TARGET (e.g. "302", "307", "34").
- PROMPT and TARGET may start or end in the middle of a sentence.
- If TARGET contains no specifics, answer {"specifics": []}."""

USER = """PROMPT:
<<<
{prompt}
>>>

TARGET:
<<<
{target}
>>>

List every specific in TARGET (not in PROMPT) with its type, evidence from PROMPT and label."""

# one worked example (synthetic text), shown as a previous user/assistant turn
EXAMPLE_PROMPT = ("the prosecution case is that on the night of the occurrence the appellant Ram Lal, along with the "
                  "co-accused, assaulted the deceased with a lathi near the village well at Bassi. The Sessions Judge, "
                  "Jaipur convicted the appellant under Section 302 read with Section 34 IPC and sentenced him to "
                  "imprisonment for life. P.W.4 Mohan Singh, the brother of the deceased, deposed that he had seen the "
                  "appellant running away from the spot. The appellant challenged the conviction before the High Court")
EXAMPLE_TARGET = ("which affirmed the conviction of Ramlal under 302/34 IPC. Before us it is urged that PW-4 is an "
                  "interested witness and that the F.I.R. lodged on 14.06.1996 at Police Station Bassi was delayed. "
                  "Shri Ashok Kumar, learned counsel for the appellant, relied on the decision of this Court in State of "
                  "Rajasthan v. Kishore, (1996) 8 SCC 217, and on the evidence of PW-7 Gopal, Ex.P-12 and a sum of "
                  "Rs.5,000 recovered from the appellant.")
EXAMPLE_ANSWER = json.dumps({"specifics": [
    {"text": "Ramlal", "type": "name", "evidence": "Ram Lal", "label": "supported"},
    {"text": "302/34 IPC", "type": "section", "evidence": "Section 302 read with Section 34 IPC", "label": "supported"},
    {"text": "PW-4", "type": "witness", "evidence": "P.W.4", "label": "supported"},
    {"text": "14.06.1996", "type": "date", "evidence": None, "label": "unsupported"},
    {"text": "Police Station Bassi", "type": "place", "evidence": None, "label": "inferable"},
    {"text": "Ashok Kumar", "type": "name", "evidence": None, "label": "unsupported"},
    {"text": "State of Rajasthan v. Kishore", "type": "citation", "evidence": None, "label": "unsupported"},
    {"text": "(1996) 8 SCC 217", "type": "citation", "evidence": None, "label": "unsupported"},
    {"text": "PW-7", "type": "witness", "evidence": None, "label": "unsupported"},
    {"text": "Gopal", "type": "name", "evidence": None, "label": "unsupported"},
    {"text": "Ex.P-12", "type": "exhibit", "evidence": None, "label": "unsupported"},
    {"text": "Rs.5,000", "type": "amount", "evidence": None, "label": "unsupported"}]})
# default separators ", " / ": " = exactly the layout xgrammar emits with disable-any-whitespace (free whitespace made
# Qwen emit runaway whitespace; a compact example under disable-any-whitespace pushed it off-distribution)

# generic legal phrases never counted as specifics (on top of the Phase 1 NER GENERIC set)
STOPLIST = set(GENERIC) | {
    "hon'ble", "honble", "hon'ble court", "this tribunal", "the bench", "learned judge", "learned single judge",
    "learned trial judge", "learned magistrate", "learned advocate", "learned counsel for the appellant",
    "learned counsel for the respondent", "learned counsel for the petitioner", "the said act", "said act",
    "the rules", "rules", "the code", "sections", "articles", "rule", "clause", "sub-section", "the petition",
    "the appeal", "the writ petition", "writ petition", "appeal", "petition", "the order", "order", "judgment",
    "the judgment", "the impugned order", "impugned order", "the impugned judgment", "the deceased", "deceased",
    "the prosecution", "prosecution", "the complainant", "complainant", "the witness", "witness", "the accused",
    "the assessee", "assessee", "the revenue", "revenue", "the department", "department", "the plaintiff",
    "plaintiff", "the defendant", "defendant", "the applicant", "applicant", "the employer", "the workman",
    "the bank", "the company", "the corporation", "the board", "the authority", "the landlord", "the tenant",
    "ld. counsel", "ld. ar", "ld. dr", "ld. cit(a)", "the ld. cit(a)", "ao", "the ao", "the assessing officer",
    "assessing officer", "cit(a)", "the cit(a)", "the high court", "the district court", "district court",
    "family court", "the family court", "labour court", "the labour court", "lower court", "the lower court",
    "appellate court", "the appellate court", "first appellate court", "the first appellate court",
    "executing court", "the executing court", "state of", "union", "the union"}
ROLE_RE = re.compile(
    r"^(?:the |said |learned |ld\.? )*(?:appellants?|respondents?|petitioners?|accused|complainants?|plaintiffs?|"
    r"defendants?|applicants?|opposite part(?:y|ies)|assessees?|claimants?|appellee|objectors?|"
    r"judgment[- ]debtors?|decree[- ]holders?|counsel|advocates?|witnesses|prosecution witness(?:es)?|"
    r"defence witness(?:es)?)(?:\s*(?:no|nos)\.?\s*\d+(?:\s*(?:,|and|&|to)\s*\d+)*)?$")


# ----------------------------------------------------------------------------- pair ids / guard

def pair_id(r):
    """Stable id of an SFT pair record (build_pairs.py fields): split|source|doc_idx|part|cut_word."""
    return f"{r['split']}|{r['source']}|{r['doc_idx']}|{r['part']}|{r['cut_word']}"


def _shingles(words, n=30):
    return {hashlib.md5(" ".join(words[i:i + n]).encode()).digest()[:8] for i in range(0, max(1, len(words) - n + 1))}


class HeldoutGuard:
    """Refuses (AssertionError) to send anything but an allowed SFT pair to the LLM.
    allowed_ids: pair ids of the SFT pair files; forbidden_texts: held-out + official test prompts;
    heldout_docs: {(source, doc_idx)} of the val documents behind the 990 held-out prompts."""

    def __init__(self, allowed_ids, forbidden_texts, heldout_docs=()):
        self.allowed = set(allowed_ids)
        self.heldout_docs = {(s, str(d)) for s, d in heldout_docs}
        self.bad = set()
        for t in forbidden_texts:
            self.bad |= _shingles(t.split())
        self.checked = 0

    def check(self, pid, prompt, target=""):
        split, source, doc_idx = pid.split("|")[:3]
        assert pid in self.allowed, f"pair {pid} is not an SFT pair"
        assert split in ("train", "val"), f"pair {pid}: bad split"
        assert not (split == "val" and (source, doc_idx) in self.heldout_docs), f"pair {pid} is from a held-out document"
        assert not (_shingles(prompt.split()) & self.bad), f"pair {pid} shares a 30-word passage with a held-out/test prompt"
        if target:
            assert not (_shingles(target.split()) & self.bad), f"pair {pid}: target overlaps a held-out/test prompt"
        self.checked += 1


def load_guard(project_dir, allowed_ids):
    P = lambda *a: os.path.join(project_dir, *a)
    ho = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    test = json.load(open(P("data", "Phase_1_Pretraining", "Test", "test.json")))
    stats = json.load(open(P("outputs", "phase1", "masked_sft", "heldout_stats.json")))
    texts = [v["text"] for v in ho.values()] + [v["text"] for v in test.values()]
    return HeldoutGuard(allowed_ids, texts, [(v["source"], v["doc_idx"]) for v in stats.values()])


# ----------------------------------------------------------------------------- prompt

def truncate_prompt(prompt, qwen_tok, max_tokens=MAX_PROMPT_QWEN_TOKENS):
    """-> (prompt text shown to the model, truncated?) keeping the LAST max_tokens Qwen tokens."""
    enc = qwen_tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= max_tokens:
        return prompt, False
    start = enc["offset_mapping"][-max_tokens][0]
    return prompt[start:], True


def messages(prompt_shown, target):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(prompt=EXAMPLE_PROMPT, target=EXAMPLE_TARGET)},
            {"role": "assistant", "content": EXAMPLE_ANSWER},
            {"role": "user", "content": USER.format(prompt=prompt_shown, target=target)}]


# ----------------------------------------------------------------------------- validation

def is_generic(text):
    k = ner_norm(text)
    return (not k) or k in STOPLIST or bool(ROLE_RE.match(k))


def find_spans(text, target):
    """All non-overlapping occurrences of `text` in `target`, not inside a longer word (a boundary is required on a
    side only where `text` starts/ends with a word character). -> (spans, found_anywhere)."""
    if not text:
        return [], False
    pat = (r"(?<!\w)" if re.match(r"\w", text[0]) else "") + re.escape(text) + (r"(?!\w)" if re.match(r"\w", text[-1]) else "")
    spans = [m.span() for m in re.finditer(pat, target)]
    return spans, (text in target)


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


PRIORITY = {"supported": 2, "inferable": 1, "unsupported": 0}


def validate_items(raw_items, prompt_shown, target):
    """raw_items: the model's "specifics" list. -> (items, counts, dropped). items: dicts text/type/label/evidence/
    spans/evidence_ok (evidence verified verbatim in prompt_shown) in first-seen order, one per distinct text;
    dropped: the items removed (text/label/type/why)."""
    c = dict(raw=0, bad_item=0, empty=0, not_verbatim=0, no_word_match=0, stoplist=0, evidence_fail=0,
             duplicate=0, label_conflict=0, kept=0, evidence_in_target_only=0)
    by_text, dropped = {}, []
    prompt_n = _norm_ws(prompt_shown)
    for it in raw_items if isinstance(raw_items, list) else []:
        c["raw"] += 1
        if not (isinstance(it, dict) and isinstance(it.get("text"), str) and it.get("label") in LABELS
                and it.get("type") in TYPES):
            c["bad_item"] += 1
            continue
        text = it["text"].strip(" \t\n,;:")
        if not text:
            c["empty"] += 1
            continue
        spans, anywhere = find_spans(text, target)
        if not spans:
            why = "no_word_match" if anywhere else "not_verbatim"
            c[why] += 1
            dropped.append(dict(text=text, label=it["label"], type=it["type"], why=why))
            continue
        if is_generic(text):
            c["stoplist"] += 1
            dropped.append(dict(text=text, label=it["label"], type=it["type"], why="stoplist"))
            continue
        label, ev = it["label"], it.get("evidence")
        ev = ev.strip() if isinstance(ev, str) else None
        ev_ok = bool(ev) and _norm_ws(ev) in prompt_n
        if label == "supported" and not ev_ok:
            c["evidence_fail"] += 1
            if ev and _norm_ws(ev) in _norm_ws(target):
                c["evidence_in_target_only"] += 1
            label = "unsupported"
        item = dict(text=text, type=it["type"], label=label, evidence=ev or None, evidence_ok=ev_ok,
                    orig_label=it["label"], spans=spans)
        if text in by_text:
            c["duplicate"] += 1
            old = by_text[text]
            if old["label"] != label:
                c["label_conflict"] += 1
            if PRIORITY[label] > PRIORITY[old["label"]]:
                by_text[text] = item
            continue
        by_text[text] = item
    items = list(by_text.values())
    c["kept"] = len(items)
    return items, c, dropped


def parse_answer(raw_text, salvage=False):
    """-> list of raw items, or None if the answer is not valid JSON of the expected shape.
    salvage: for an answer cut off at max_tokens, keep the items before the cut (tries closing the array after
    each of the last 200 "}" from the end)."""
    def ok(obj):
        return obj["specifics"] if isinstance(obj, dict) and isinstance(obj.get("specifics"), list) else None
    try:
        return ok(json.loads(raw_text))
    except (json.JSONDecodeError, TypeError):
        pass
    if not salvage or not isinstance(raw_text, str):
        return None
    ends = [i for i, ch in enumerate(raw_text) if ch == "}"][::-1][:200]
    for i in ends:
        try:
            return ok(json.loads(raw_text[:i + 1] + "]}"))
        except json.JSONDecodeError:
            continue
    return None


# ----------------------------------------------------------------------------- resumable store

def split_files(labels_dir, split):
    """All JSONL stores of a split: <split>.jsonl (pilot / single run) + <split>.<tag>.jsonl (one per labeling process)."""
    import glob
    return sorted(glob.glob(os.path.join(labels_dir, f"{split}.jsonl")) + glob.glob(os.path.join(labels_dir, f"{split}.*.jsonl")))


def load_done_dir(labels_dir, split):
    done = set()
    for f in split_files(labels_dir, split):
        done |= load_done(f)
    return done


def load_records_dir(labels_dir, split):
    out = {}
    for f in split_files(labels_dir, split):
        for k, v in load_records(f).items():
            out.setdefault(k, v)
    return out


def load_done(path):
    """pair ids already in the JSONL store (a torn last line from an interrupted write is ignored)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["pair_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def load_records(path):
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["pair_id"], r)        # first record wins (there are never duplicates by construction)
    return out


def repair_tail(path):
    """Drop a torn final line (interrupted write) so that appends start on a fresh line."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def todo(pairs_with_ids, done):
    """Pairs (pid, record) not yet labeled, in input order, each pid once."""
    seen, out = set(done), []
    for pid, r in pairs_with_ids:
        if pid not in seen:
            seen.add(pid)
            out.append((pid, r))
    return out
