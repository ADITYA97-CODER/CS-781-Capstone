"""Rule-based cleaning of training documents (section 0 of the masked-task-SFT plan).

    clean_document(text, source) -> list[str]

1. Split run-on records. A document is ended (and a new one started) at a marker only when ALL hold:
   - marker: "ITEM NO", "RECORD OF PROCEEDINGS", "S U P R E M E", "IN THE SUPREME COURT OF INDIA"
     (any source); for high_courts also "IN THE HIGH COURT OF", but only after 90% of the document (chars).
     Markers closer than CLUSTER_CHARS to the previous marker form one cluster, evaluated at its first marker.
   - a judge signature AND a place/date line both lie within the WINDOW_TOKENS tokens before the marker
     (and after any previous split);
   - the marker starts at most GAP_TOKENS tokens after the end of the last signature/date match
     (allows "REPORTABLE", "(FOR JUDGMENT)", "Page 23 of 23" between them);
   - not inside a quotation: no "quote"/"quoted", ":-" or opening quote mark in the QUOTE_TOKENS tokens
     before the marker.
   The cut is placed right after the last signature/date match, so the first part ends with the
   signature/date block and the second part starts with whatever follows (header / gap text).
2. Remove trailing typist initials (/[a-z]{2,})+ : only when they end the document (nothing but
   whitespace or "/" after them), are preceded by whitespace, and start in the last 5% (chars).
3. Flag (no change) documents with Kruti-Dev garbled Hindi.

No other text is changed. analyze_document() returns the same output plus a record of every action /
rejection for reporting.
"""
import os
import re

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOKENIZER_DIR = os.path.join(PROJECT_DIR, "artifacts", "tokenizer", "legal_bpe_32k")

WINDOW_TOKENS = 200
GAP_TOKENS = 25
QUOTE_TOKENS = 50
HC_MIN_FRAC = 0.90
INITIALS_LAST_FRAC = 0.05
CLUSTER_CHARS = 300

SC_MARKERS = re.compile(r"ITEM NO|RECORD OF PROCEEDINGS|S U P R E M E|IN THE SUPREME COURT OF INDIA")
HC_MARKER = re.compile(r"IN THE HIGH COURT OF")

_SEP = r"(?:\s+|(?<=\.)\s*)"  # name parts are separated by a space, or follow an initial's dot
_NOT_HEADER = r"(?!(?i:for|reportable|non-reportable|signed|hearing|corrected|revised|arising|with)\b)"
_NAME = (r"(?:Dr\.?\s*|Justice\s+|Mr\.?\s*|Mrs\.?\s*|Ms\.?\s*)?" + _NOT_HEADER
         + r"[A-Z][A-Za-z.'\-]*(?:" + _SEP + r"[A-Z][A-Za-z.'\-]*){0,6}")
SIGNATURE = re.compile(
    # "J. (A.K. SIKRI)", "J. [R. BANUMATHI]", "CJI. [Dr D Y Chandrachud]" (signature line flattened before the name)
    r"(?P<jb>\b(?:J|CJI|CJ)\s*\.?\s*[\(\[]\s*" + _NAME + r"\s*[\)\]]?)"
    # "R. BANUMATHI, J." / "Arun Mishra J."
    r"|(?P<nj>\b[A-Z][A-Za-z.'\-]*(?:" + _SEP + r"[A-Z][A-Za-z.'\-]*){0,5}\s*,?\s*(?:J|CJI|CJ)\s*\.(?![A-Za-z0-9]))"
    # "(Arun Mishra)" / "(S.B. SINHA)": capitalised name of >= 2 parts in brackets
    r"|(?P<br>[\(\[]\s*(?:Dr\.?\s*|Justice\s+)?" + _NOT_HEADER + r"[A-Z][A-Za-z.'\-]*(?:" + _SEP
    + r"[A-Z][A-Za-z.'\-]*){1,6}\s*[\)\]])"
)
# Andhra Pradesh High Court form: all-caps name followed by ", J" / ", CJ", period optional (high_courts only)
AP_SIGNATURE = re.compile(r"\b[A-Z][A-Z.'\-]+(?:\s+[A-Z][A-Z.'\-]*){0,5}\s*,\s*C?J\b\.?")
# leading junk removed from documents created by a split (only within their first LEAD_CHARS characters)
LEAD_CHARS = 100
LEAD_PATTERNS = [
    ("page_footer", re.compile(r"(?i:page)\s*\d+\s*of\s*\d+\b\.?")),
    ("typist_mark", re.compile(r"\([A-Za-z]{2,8}\)")),                                  # (sks) (BC) (Hem)
    ("typist_mark", re.compile(r"'[a-z]{2,12}(?:/[a-z]{2,12})*'")),                       # 'dalbir/rana'
    ("typist_mark", re.compile(r"[a-z]{1,12}(?:/[a-z]{1,12})*(?=\s*\*?\s*[A-Z0-9\"'(\[])")),  # vikrant / nsk * / s
    ("stray_punct", re.compile(r"[,;)*\]]+")),
]
_MONTH = (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE = (rf"(?:{_MONTH}\s*\d{{1,2}}(?:st|nd|rd|th)?\s*[,.]?\s*(?:19|20)\d\d"
         rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s*(?:day\s+of\s+)?{_MONTH}\s*[,.]?\s*(?:19|20)\d\d"
         r"|\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*(?:19|20)?\d\d)")
_PLACE = r"(?P<place>(?:NEW\s+DELHI|New\s+Delhi|Dated?\s*[:.\-]?|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s*[,;:.]?\s*)"
PLACE_DATE = re.compile(rf"{_PLACE}?(?i:{_DATE})\.?"
                        rf"|(?P<place2>(?i:new\s+delhi)\s*[,;:.]?\s*)(?i:{_MONTH}\s*,?\s*(?:19|20)\d\d)\.?")
# words that belong to the signature block and follow the names: the cut is moved past them
SIG_TAIL = re.compile(r"\s*(?:,?\s*(?:\([A-Za-z]{2,8}\)\s*)?(?i:judge)\b\.?|(?i:court\s+master|ps\s+to\s+registrar|branch\s+officer|ar-cum-ps|assistant\s+registrar"
                      r"|deputy\s+registrar|sr\.?\s*p\.?a\.?|p\.?s\.?)\b[.,]?"
                      r"|[;,]\s*(?i:new\s+delhi)\b\.?"  # "October 3, 2023 ; New Delhi."
                      r"|[\(\[]\s*" + _NOT_HEADER + r"[A-Z][A-Za-z.'\-]*(?:" + _SEP + r"[A-Z][A-Za-z.'\-]*){1,6}\s*[\)\]]"  # 2nd judge after the date
                      r"|[\(\[]?(?i:(?:the\s+)?signed\s+(?:reportable\s+|non-reportable\s+)?(?:order|judgment)s?\s+(?:is|are)\s+placed\s+on\s+(?:the\s+)?file)\.?[\)\]]?\.?)")
QUOTE_WORD = re.compile(r"\bquot(?:e|ed|ing)\b|:-", re.I)
OPEN_QUOTE = re.compile(r"[“‘„«]|(?:^|(?<=[\s(:\[]))[\"'](?=\S)")
CLOSE_QUOTE = re.compile(r"[”’»]|(?<=\S)[\"'](?=[\s.,;:)\]\[]|$)")
INITIALS = re.compile(r"(?<!\S)((?:/[a-z]{2,})+)/?\s*\Z")

# Kruti-Dev (legacy Hindi font) words rendered as ASCII: के में की का है और कि से को भी यह था किया गया पर एक हैं तथा ...
KRUTI = {"ds", "esa", "dh", "dk", "gS", "gSA", "vkSj", "fd", "ls", "dks", "Hkh", ";g", "Fkk", "FkkA", "fd;k",
         "x;k", "ij", ",d", "gSa", "rFkk", "fdlh", "mlds", "ftl", "bl", ";k", "dj", "djus", "fy,", "gq,", "tks"}
KRUTI_MIN, KRUTI_FRAC = 10, 0.02
GARBLED_FRAC = 0.20  # documents with >= 20% Kruti-Dev words are dropped

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from transformers import PreTrainedTokenizerFast
        _tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    return _tok


def _char_back(text, end, n_tokens, floor=0):
    """Char offset where the n_tokens tokens ending at `end` begin (>= floor)."""
    lo = max(floor, end - 12 * n_tokens)
    offs = _tokenizer()(text[lo:end], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offs) <= n_tokens:
        return lo
    return lo + offs[-n_tokens][0]


def _n_tokens(s):
    return len(_tokenizer()(s, add_special_tokens=False)["input_ids"]) if s.strip() else 0


def kruti_frac(text):
    words = text.split()
    return sum(w.strip(".,()\"'|") in KRUTI for w in words) / max(1, len(words))


def is_garbled(text):
    return is_kruti_dev(text) and kruti_frac(text) >= GARBLED_FRAC


def is_kruti_dev(text):
    if " ds " not in text and " esa " not in text and " gS" not in text and " dh " not in text:
        return False  # fast path: every Kruti-Dev text has at least one of these
    words = text.split()
    if not words:
        return False
    hits = sum(w.strip(".,()\"'|") in KRUTI for w in words)
    return hits >= KRUTI_MIN and hits / len(words) >= KRUTI_FRAC


_SIG_CONTEXT = re.compile(r"\s*(?:,?\s*J\.|(?i:judge|member|chairman|chairperson|president|vice[- ]chairman|court\s+master|ps\s+to)\b)")


def _valid_signature(m, text, offset):
    """Reject header fields that look like bracketed names: "(EARLY HEARING)", "A.J.C Bose Road"."""
    g = m.group()
    if not re.search(r"[A-Za-z]{3}", re.sub(r"\b(?:CJI|CJ)\b", "", g)):
        return False  # "A.J." is not a name
    if m.group("br"):
        inner = g.strip("()[] ")
        if "." in inner or re.search(r"[a-z]", inner):
            return True  # "(Arun Mishra)", "(S.B. SINHA)"
        return bool(_SIG_CONTEXT.match(text, offset + m.end()))  # "(KALYANI GUPTA) Court Master"
    return True


def _date_kind(m):
    pl = m.group("place") or m.group("place2")
    if not pl:
        return "bare"
    return "label" if re.match(r"Dated?\b", pl) else "place"


def _best_kind(dates):
    kinds = {k for *_, k in dates}
    return next(k for k in ("place", "label", "bare") if k in kinds)


def _strip_leading(doc):
    """Remove page footers / typist marks / stray punctuation at the start of a split-off document (first LEAD_CHARS chars)."""
    removed, pos = [], 0
    while True:
        while pos < len(doc) and doc[pos].isspace():
            pos += 1
        for kind, rx in LEAD_PATTERNS:
            m = rx.match(doc, pos)
            if m and m.end() > pos and m.end() <= LEAD_CHARS:
                removed.append((kind, m.group()))
                pos = m.end()
                break
        else:
            return doc[pos:], removed


def _markers(text, source):
    if source == "supreme_court":
        ms = [(m.start(), m.group()) for m in SC_MARKERS.finditer(text)]
    elif source == "high_courts":
        ms = [(m.start(), m.group()) for m in HC_MARKER.finditer(text, int(HC_MIN_FRAC * len(text)))]
    else:  # tribunals / acts: no splitting (quoted orders and order-sheet headers stay as they are)
        return []
    ms.sort()
    clusters, last_end = [], -10 ** 9
    for p, g in ms:
        if p - last_end > CLUSTER_CHARS:
            clusters.append((p, g))
        last_end = p + len(g)
    return clusters


def analyze_document(text, source):
    """-> (docs, info). info: splits [dict], rejects [dict], initials dict|None, kruti bool."""
    info = dict(splits=[], rejects=[], initials=None, kruti=is_kruti_dev(text))
    cuts, floor = [], 0
    for p, marker in _markers(text, source):
        if p <= floor:
            continue
        rej = dict(pos=p, marker=marker)
        w0 = _char_back(text, p, WINDOW_TOKENS, floor)
        win = text[w0:p]
        sigs = [(w0 + m.start(), w0 + m.end()) for m in SIGNATURE.finditer(win) if _valid_signature(m, text, w0)]
        ap_only = False
        if source == "high_courts":
            ap = [(w0 + m.start(), w0 + m.end()) for m in AP_SIGNATURE.finditer(win)]
            ap_only = bool(ap) and not sigs
            sigs += ap
        dates = [(w0 + m.start(), w0 + m.end(), _date_kind(m)) for m in PLACE_DATE.finditer(win)]
        q0 = _char_back(text, p, QUOTE_TOKENS, floor)
        qm = QUOTE_WORD.search(text, q0, p)
        if not qm:  # an opening quote counts only if no closing quote follows it before the marker
            for oq in OPEN_QUOTE.finditer(text, q0, p + 1):  # +1: a quote right before the marker needs lookahead
                if not CLOSE_QUOTE.search(text, oq.end(), p):
                    qm = oq
                    break
        reasons = []
        if not sigs:
            reasons.append("no_signature")
        if not dates:
            reasons.append("no_date")
        anchor = max([e for _, e in sigs] + [e for _, e, _ in dates], default=None)
        while anchor is not None and (t := SIG_TAIL.match(text, anchor, p)) and t.end() > anchor:
            anchor = t.end()
        if anchor is not None:
            gap = _n_tokens(text[anchor:p])
            if gap > GAP_TOKENS:
                reasons.append(f"gap_{gap}_tokens")
        if qm:
            reasons.append(f"quote({qm.group()!r})")
        if reasons:
            info["rejects"].append(dict(rej, reasons=reasons))
            continue
        # cut after the anchor, swallowing following whitespace into nothing
        cut = anchor
        info["splits"].append(dict(rej, cut=cut, ap_signature_only=ap_only, date_kind=_best_kind(dates),
                                   rel_pos=round(p / len(text), 3)))
        cuts.append(cut)
        floor = cut
    # trailing typist initials: must end the original document, start in its last 5%, after any cut
    end = len(text)
    m = None if source == "acts" else INITIALS.search(text, int((1 - INITIALS_LAST_FRAC) * len(text)))
    if m and m.start() >= (1 - INITIALS_LAST_FRAC) * len(text) and m.start() >= (cuts[-1] if cuts else 0):
        info["initials"] = dict(pos=m.start(), removed=m.group(1))
        end = m.start()
    parts, prev = [], 0
    for c in cuts + [end]:
        parts.append(text[prev:c].strip())
        prev = c
    info["lead_removed"] = []
    for i in range(1, len(parts)):  # documents created by a split only
        parts[i], removed = _strip_leading(parts[i])
        info["lead_removed"] += removed
    parts = [x for x in parts if x]
    return parts, info


def clean_document(text, source, drop_garbled=True):
    """-> list of cleaned documents; [] for a >= 20% Kruti-Dev garbled document (unless drop_garbled=False)."""
    if drop_garbled and is_garbled(text):
        return []
    return analyze_document(text, source)[0]
