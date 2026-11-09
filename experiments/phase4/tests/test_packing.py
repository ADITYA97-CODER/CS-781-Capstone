"""Packed FlexAttention encoder == Phase-2 dense bidirectional encoder run on each chunk alone (chunk mean vectors)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cjpe_data as D  # noqa: E402
from cjpe_model import DocClassifier, load_backbone  # noqa: E402

sys.path.insert(0, os.path.join(D.ROOT, "phase2"))
from ner_model import Tagger  # noqa: E402

dev = torch.device("cuda:0")
for ck in ["runs/sft_archE_v2_upweight/ckpt_best.pt", "runs/archF_nodiff_145M/ckpt_best.pt"]:
    bb, cfg = load_backbone(os.path.join(D.ROOT, ck))
    model = DocClassifier(bb, n_pos=1, dropout=0.0).to(dev).eval()
    ref = Tagger(model.backbone, 2, bidirectional=True, dropout=0.0).eval()
    ids, y, toks = D.load_tokens("dev_clean")
    rng = np.random.RandomState(0)
    for view, L, K in [("last", 1024, 1), ("hier", 512, 4), ("headtail", 1024, 1)]:
        docs = rng.choice(len(toks), 12, replace=False)
        # include some short documents so that rows hold several chunks
        toks_s = [toks[i][: rng.randint(100, 3000)] for i in docs]
        chunks = [D.doc_chunks(t, view, L, K) for t in toks_s]
        lens = [len(t) for t in toks_s]
        mbs = D.micro_batches(chunks, list(range(len(chunks))), 4, L)
        err = 0.0
        with torch.no_grad():
            for mb in mbs:
                b = D.pack(mb, chunks, lens, 4, L, 1, dev)
                masks = __import__("cjpe_model").segment_masks(b["seg"], model.window)
                h = model.encode(b["ids"], b["pos"], masks).float()
                for (d, place) in mb:
                    for (c, _), (r, o) in zip(chunks[d], place):
                        got = h[r, o:o + len(c)]
                        x = torch.from_numpy(c)[None].to(dev)
                        want = ref.hidden(x, torch.ones_like(x, dtype=torch.bool))[0].float()
                        err = max(err, (got - want).abs().max().item() / want.abs().max().item())
        print(f"{ck.split('/')[1]} {view} L{L} K{K}: {len(mbs)} micro-batches, max rel err {err:.2e}")
        assert err < 1e-3, err
print("OK")
