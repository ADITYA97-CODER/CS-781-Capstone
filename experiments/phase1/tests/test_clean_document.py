"""Tests for phase1/corpus/clean_document.py.   python phase1/tests/test_clean_document.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase1.corpus.clean_document import analyze_document, clean_document, is_kruti_dev  # noqa: E402

BODY = "The appellant was convicted under Section 302 of the Indian Penal Code by the trial court. " * 40
SIG = "The appeal is allowed. No costs. J. (UDAY UMESH LALIT) J. (S. RAVINDRA BHAT) New Delhi, March 07, 2022. "
ROP = ("ITEM NO.10 COURT NO.2 SECTION II-C S U P R E M E C O U R T O F I N D I A RECORD OF PROCEEDINGS "
       "Special Leave Petition (Crl.) No.11390 of 2019 ABHISHEK KUMAR Appellant(s) VERSUS STATE OF DELHI. ") + BODY[:600]


def test_sc_split():
    docs, info = analyze_document(BODY + SIG + ROP, "supreme_court")
    assert len(docs) == 2, info
    assert docs[0].endswith("New Delhi, March 07, 2022.") and docs[1].startswith("ITEM NO.10"), docs
    assert len(info["splits"]) == 1 and info["splits"][0]["date_kind"] == "place"


def test_sc_split_with_gap_text():
    docs = clean_document(BODY + SIG + "(FOR JUDGMENT) (REPORTABLE) " + ROP, "supreme_court")
    assert len(docs) == 2 and docs[1].startswith("(FOR JUDGMENT) (REPORTABLE) ITEM NO"), docs[1][:80]
    docs = clean_document(BODY + SIG + "REPORTABLE IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION "
                          + BODY[:500], "supreme_court")
    assert len(docs) == 2 and docs[1].startswith("REPORTABLE IN THE SUPREME COURT OF INDIA")


def test_reject_without_signature_or_date():
    t = BODY + "The appeal is allowed. No costs. " + ROP
    docs, info = analyze_document(t, "supreme_court")
    assert docs == [t.strip()] and info["rejects"] and "no_signature" in info["rejects"][0]["reasons"]
    t = BODY + "The appeal is allowed. J. (A.K. SIKRI) J. (ASHOK BHUSHAN) " + ROP
    docs, info = analyze_document(t, "supreme_court")
    assert docs == [t.strip()] and "no_date" in info["rejects"][0]["reasons"]


def test_reject_quoted_order():
    # tribunal quoting a Supreme Court record of proceedings: must not be split
    for q in ['which we quote:- "', 'as quoted below: ', 'reads as under:- ', 'as follows: "', "reads: “"]:
        t = BODY + SIG + "The matter is covered by an order of the Apex Court " + q + ROP
        docs, info = analyze_document(t, "supreme_court")
        assert docs == [t.strip()], (q, info)
        assert any("quote" in r for r in info["rejects"][0]["reasons"]), info


def test_closed_quote_does_not_block():
    t = BODY + 'it is humility that forms the "foundation of regard"[28]. ' + SIG + ROP
    assert len(clean_document(t, "supreme_court")) == 2


def test_no_split_for_tribunals_and_acts():
    t = BODY + SIG + ROP
    for src in ["tribunals", "acts"]:
        assert clean_document(t, src) == [t.strip()]


def test_order_sheet_header_not_a_signature():
    t = (BODY + "Kolkata ORDER SHEET COURT NO. : 2 11.12.2020 O.A./350/1612/2019 BINAPANI MIDDYA -V/S- (SB) (EARLY HEARING) "
         "(MENTIONED) M/O DEFENCE ITEM NO: 1 FOR APPLICANTS(S) Adv. : Ms. M. Patra " + BODY[:300])
    docs, info = analyze_document(t, "supreme_court")
    assert docs == [t.strip()] and "no_signature" in info["rejects"][0]["reasons"]
    t = BODY + "234/4 A.J.0 Bose Road Nizam Palace Kolkata 10.08.2018 ITEM NO:5 FOR APPLICANTS " + BODY[:300]
    assert clean_document(t, "supreme_court") == [t.strip()]


def test_hp_judge_tail_and_month_year_date():
    t = (BODY * 3 + "both the appeals are dismissed. April 24, 2010 (Kuldip Singh) , (Hem) Judge. "
         "IN THE HIGH COURT OF HIMACHAL PRADESH, SHIMLA. Cr. Appeal No.228 of 2009")
    docs = clean_document(t, "high_courts")
    assert len(docs) == 2 and docs[0].endswith("(Hem) Judge."), docs[0][-60:]
    t = BODY + "dismissed with costs. J. (S.B. SINHA) J. (V.S. SIRPURKAR) New Delhi April, 2008 " + ROP
    assert len(clean_document(t, "supreme_court")) == 2


def test_reject_long_gap():
    t = BODY + SIG + BODY[:400] + ROP
    docs, info = analyze_document(t, "supreme_court")
    assert docs == [t.strip()] and any(r.startswith("gap_") for r in info["rejects"][0]["reasons"])


def test_hc_marker_only_after_90pct():
    hc = SIG + "IN THE HIGH COURT OF DELHI AT NEW DELHI W.P.(C) 123/2020 "
    early = BODY + hc + BODY * 3
    assert clean_document(early, "high_courts") == [early.strip()]
    late = BODY * 3 + hc + "Date of decision: 3rd September, 2012."
    docs = clean_document(late, "high_courts")
    assert len(docs) == 2 and docs[1].startswith("IN THE HIGH COURT OF DELHI")
    # the HC marker is not a split marker for other sources
    assert clean_document(late, "tribunals") == [late.strip()]


def test_unchanged_text():
    for t, src in [
        ("CUSTOMS, EXCISE AND SERVICE TAX APPELLATE TRIBUNAL West Zonal Bench AHMEDABAD COURT NO. I Appeal No. C/464/2009 " + BODY, "tribunals"),
        (BODY + "ADDITIONAL SESSIONS COURT (FAST TRACK COURT NO.1) THRISSUR, CRIME NO.29/08 " + BODY, "high_courts"),
        (BODY + SIG + "COURT NO. 3 " + BODY, "supreme_court"),
        (BODY + "Concerned Appellate Authorities /bodies", "acts"),
        (BODY + "VIDE ITEM NO.2. //TRUE COPY// P.A. TO JUDGE ANTONY DOMINIC, J. " + BODY, "high_courts"),
        (BODY + "the petitioner and/or his agents shall pay the dues and/or", "high_courts"),
        (BODY + "Rs. 500/- (rupees five hundred only) /-", "tribunals"),
    ]:
        docs, info = analyze_document(t, src)
        assert docs == [t.strip()], (t[-120:], info)
        assert info["initials"] is None and not info["splits"]


def test_trailing_initials():
    t = BODY + "No order as to costs. (Aradhana Johri) (Justice L. Narasimha Reddy) Member (A) Chairman /lg/pj/jyoti/mbt/"
    docs, info = analyze_document(t, "tribunals")
    assert docs == [t[:t.index(" /lg/")]] and info["initials"]["removed"] == "/lg/pj/jyoti/mbt"
    t2 = BODY + "Member (A) August 2, 2018 /sunil/  "
    assert clean_document(t2, "tribunals") == [t2[:t2.index(" /sunil")]]
    # not trailing (page footer after it): unchanged
    t3 = BODY + "MEMBER (J) /ksk/ 18 OA.No.170/01006/2019/CAT/BANGALORE"
    assert clean_document(t3, "tribunals") == [t3]
    # initials that are not in the last 5% of a short document's characters are kept
    t5 = "x " * 5 + "/ab/cd/"  # match starts at 10/17 = 59% < 95%
    assert clean_document(t5, "tribunals") == [t5.strip()]


def test_cut_after_court_master_tail():
    t = (BODY + "Date : 25/02/2009 This petition was called on for hearing today. The appeal is allowed in terms of the "
         "signed order. (Sukhbir Paul Kaur) (Vinod Kulvi) Court Master Court Master (Signed Order is placed on the file) "
         "IN THE SUPREME COURT OF INDIA CRIMINAL APPELLATE JURISDICTION " + BODY[:300])
    docs, info = analyze_document(t, "supreme_court")
    assert len(docs) == 2 and docs[0].endswith("(Signed Order is placed on the file)"), docs[0][-80:]
    assert docs[1].startswith("IN THE SUPREME COURT OF INDIA") and info["splits"][0]["date_kind"] == "label"


def test_tail_variants():
    for mid, end in [
        ("Date: 08/10/2009 The appeal is dismissed. (S. Thapar) (Madhu Saxena) PS to Registrar Court Master The signed order is placed on the file. ",
         "The signed order is placed on the file."),
        ("J. (A.S. BOPANNA) J. (SANJAY KUMAR) October 3, 2023 ; New Delhi. ", "; New Delhi."),
        ("J. [JAGDISH SINGH KHEHAR] NEW DELHI; J. JULY 23, 2015. [ADARSH KUMAR GOEL] ", "[ADARSH KUMAR GOEL]"),
    ]:
        docs = clean_document(BODY + mid + ROP, "supreme_court")
        assert len(docs) == 2 and docs[0].endswith(end) and docs[1].startswith("ITEM NO"), (docs[0][-80:], docs[1][:40])


def test_ap_signature_high_courts_only():
    t = (BODY * 3 + "The writ appeal is dismissed. No costs. ARUP KUMAR GOSWAMI, CJ NINALA JAYASURYA, J Date: 05.10.2021 "
         "IN THE HIGH COURT OF ANDHRA PRADESH: AMARAVATI HON'BLE MR. JUSTICE ARUP KUMAR GOSWAMI")
    docs, info = analyze_document(t, "high_courts")
    assert len(docs) == 2 and info["splits"][0]["ap_signature_only"], info
    # still needs the date
    t2 = t.replace("Date: 05.10.2021 ", "")
    assert clean_document(t2, "high_courts") == [t2.strip()]
    # not used for the Supreme Court
    t3 = BODY + "Appeal dismissed. ARUP KUMAR GOSWAMI, CJ New Delhi, March 07, 2022. " + ROP
    assert clean_document(t3, "supreme_court") == [t3.strip()]


def test_leading_junk_in_new_docs():
    for junk, kinds in [("Page 118 of 118 ", ["page_footer"]), ("(sks) 8 ", ["typist_mark"]), ("vikrant ", ["typist_mark"]),
                        ("'dalbir/rana' ", ["typist_mark"]), (", (Hem) ", ["stray_punct", "typist_mark"]), ("nsk * ", ["typist_mark", "stray_punct"])]:
        t = BODY + SIG + junk + "REPORTABLE IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION " + BODY[:300]
        docs, info = analyze_document(t, "supreme_court")
        assert len(docs) == 2 and [k for k, _ in info["lead_removed"]] == kinds, (junk, info["lead_removed"])
        want = "8 REPORTABLE" if junk.startswith("(sks) 8") else "REPORTABLE"
        assert docs[1].startswith(want), (junk, docs[1][:40])
    # the first (original) document is never stripped
    t = "Page 1 of 9 " + BODY + SIG + ROP
    assert clean_document(t, "supreme_court")[0].startswith("Page 1 of 9")


def test_split_and_initials_together():
    t = BODY + SIG + ROP + " Court Master /ab/cd/"
    docs, info = analyze_document(t, "supreme_court")
    assert len(docs) == 2 and docs[1].endswith("Court Master") and info["initials"]


def test_kruti():
    hindi = "ifjoknh us foi{kh ds fo:) ;g f'kdk;r dh gS fd foi{kh us mlds lkFk /kks[kk/kM+h dh rFkk ml ij vkjksi gSA " * 10
    assert is_kruti_dev(BODY + hindi)
    assert not is_kruti_dev(BODY)
    docs, info = analyze_document(BODY + hindi, "high_courts")
    assert info["kruti"] and docs == [(BODY + hindi).strip()]
    heavy = BODY[:300] + hindi
    assert clean_document(heavy, "tribunals") == [] and clean_document(heavy, "tribunals", drop_garbled=False) == [heavy.strip()]
    assert clean_document(BODY + hindi[:200], "tribunals") == [(BODY + hindi[:200]).strip()]  # lightly garbled: kept


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            n += 1
            print("ok", k)
    print(f"{n}/{n} passed")
