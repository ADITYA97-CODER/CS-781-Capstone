"""Statistics (not content) of the 990 local held-out prompts, and the documents they come from.

  python phase1/sft/heldout_stats.py     # -> outputs/phase1/masked_sft/heldout_stats.json

Each prompt was built by filter_split.py from a val document: 512 whitespace words starting at a random word,
followed by the 128-word reference. Here every prompt is located in its val document (whitespace-normalized
text) to recover the cut position. Output: per-prompt source, val doc_idx, doc length (words), cut word index,
relative cut position, prompt length in legal_bpe_32k tokens, ends-mid-sentence flag.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
OUT = P("outputs", "phase1", "masked_sft")
SENT_END = re.compile(r"[.?!][\"'”’)\]]*$")


def ends_mid_sentence(text):
    return not SENT_END.search(text.rstrip())


def main():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(P("artifacts", "tokenizer", "legal_bpe_32k"))
    prompts = json.load(open(P("data", "corpus", "eval", "local_test.json")))
    refs = json.load(open(P("data", "corpus", "eval", "local_ref.json")))
    by_src = {}
    for k, r in refs.items():
        by_src.setdefault(r["source"], []).append(k)
    recs = {}
    for src, keys in by_src.items():
        t = pq.read_table(P("data", "corpus", "clean", f"{src}.val.parquet"), columns=["doc_idx", "text"])
        docs = [(d, " ".join(x.split())) for d, x in zip(t.column("doc_idx").to_pylist(), t.column("text").to_pylist())]
        for k in keys:
            p = prompts[k]["text"]
            hits = [(d, x) for d, x in docs if p in x]
            assert len(hits) >= 1, (k, src)
            d, x = hits[0]
            start_word = len(x[:x.index(p)].split())
            n = len(x.split())
            cut = start_word + 512
            recs[k] = dict(source=src, doc_idx=d, n_dups=len(hits), doc_words=n, cut_word=cut, rel_cut=cut / n,
                           words_left=n - cut, prompt_tokens=len(tok(p, add_special_tokens=False)["input_ids"]),
                           mid_sentence=ends_mid_sentence(p))
    os.makedirs(OUT, exist_ok=True)
    json.dump(recs, open(os.path.join(OUT, "heldout_stats.json"), "w"), indent=0)
    v = list(recs.values())
    pt = np.array([r["prompt_tokens"] for r in v])
    rc = np.array([r["rel_cut"] for r in v])
    print("prompts", len(v), "source mix", Counter(r["source"] for r in v))
    print("prompt tokens: mean %.0f p5 %.0f p50 %.0f p95 %.0f max %d" % (pt.mean(), *np.percentile(pt, [5, 50, 95]), pt.max()))
    print("mid-sentence %.3f | rel cut: mean %.3f, share >=0.9 %.3f, share >=0.8 %.3f | words left after target <=0: %d"
          % (np.mean([r["mid_sentence"] for r in v]), rc.mean(), (rc >= 0.9).mean(), (rc >= 0.8).mean(),
             sum(r["words_left"] - 128 <= 0 for r in v)))
    print("docs with >1 match:", sum(r["n_dups"] > 1 for r in v), "| distinct held-out docs:", len({(r['source'], r['doc_idx']) for r in v}))


if __name__ == "__main__":
    main()
