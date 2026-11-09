"""Extractor of specifics (dates, citations, amounts, sections, names): regex patterns plus NER helpers.
Used by the masked-SFT loss mask and by the penalty-aware reranker."""

import re

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
