"""Tests for intra-document masking (model.doc_mask) in Arch D.

Run on GPU 0 (FlexAttention needs CUDA):
  CUDA_VISIBLE_DEVICES=0 python phase1/tests/test_doc_mask.py
Tests 1-7 use a tiny random model in fp32; test 8 (timing) is phase1/tests/bench_doc_mask.py.
"""

import os
import sys

import torch
from torch.nn.attention.flex_attention import create_mask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import mask_boundary_labels  # noqa: E402
from models.arch_d_sliding import ArchDConfig, ArchDSliding  # noqa: E402
from models.layers import doc_structure, lm_loss  # noqa: E402

EOS = 0
DEV = "cuda"
ATOL = 1e-4


def tiny_model(doc_mask=True, window=32):
    torch.manual_seed(0)
    cfg = ArchDConfig(vocab_size=200, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                      max_seq_len=512, window=window, global_every=2, doc_mask=doc_mask, eos_id=EOS)
    return ArchDSliding(cfg).to(DEV).eval()


def rand_doc(n, gen):
    return torch.randint(5, 200, (n,), generator=gen)            # ids 5.. never collide with <|endoftext|>


def logits(model, row):
    with torch.no_grad():
        return model(row[None].to(DEV) if row.dim() == 1 else row.to(DEV))[0]


# ----------------------------------------------------------------------------- 1. doc_structure

def test_doc_structure():
    A1, A2, A3, B1, B2, C1 = 11, 12, 13, 21, 22, 31
    ids = torch.tensor([[A1, A2, A3, EOS, B1, B2, EOS, C1]])
    d, p = doc_structure(ids, EOS)
    assert d.tolist() == [[0, 0, 0, 0, 1, 1, 1, 2]], d
    assert p.tolist() == [[0, 1, 2, 3, 0, 1, 2, 0]], p

    batch = torch.tensor([
        [5, 6, EOS, 7, 8, 9, 10, 11],          # boundary after position 2
        [5, 6, 7, 8, 9, EOS, 10, 11],          # boundary after position 5
        [5, 6, 7, 8, 9, 10, 11, 12],           # no <|endoftext|>: one document
        [5, 6, 7, EOS, 8, 9, 10, EOS],         # ends in <|endoftext|>
        [EOS, EOS, 5, 6, 7, 8, 9, 10],         # starts with two (empty document between)
    ])
    d, p = doc_structure(batch, EOS)
    assert d.tolist() == [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1], [0] * 8,
                          [0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 2, 2, 2, 2, 2]], d
    assert p.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 0, 1], list(range(8)),
                          [0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 0, 1, 2, 3, 4, 5]], p


# ----------------------------------------------------------------------------- 2. mask pattern

def test_mask_block_diagonal():
    ids = torch.tensor([[11, 12, 13, EOS, 21, 22, EOS, 31],
                        [11, EOS, 21, 22, 23, 24, 25, 26]], device=DEV)   # rows have different boundaries
    doc_ids, _ = doc_structure(ids, EOS)
    L = ids.size(1)
    expected = torch.zeros(2, L, L, dtype=torch.bool)
    for b, blocks in enumerate([[(0, 4), (4, 7), (7, 8)], [(0, 2), (2, 8)]]):
        for s, e in blocks:
            expected[b, s:e, s:e] = torch.ones(e - s, e - s, dtype=torch.bool).tril()

    model = tiny_model(window=3)
    local_mod, global_mod = model.doc_mask_mods(doc_ids, window=100)       # window larger than L
    flex_dense = create_mask(global_mod, 2, None, L, L, device=DEV)[:, 0].cpu()
    assert torch.equal(flex_dense, expected), flex_dense
    assert torch.equal(create_mask(local_mod, 2, None, L, L, device=DEV)[:, 0].cpu(), expected)

    model.doc_attn_impl = "dense"                                           # fallback masks, window 3
    local_d, global_d = model.doc_block_masks(doc_ids)
    assert torch.equal(global_d[:, 0].cpu(), expected)
    i = torch.arange(L)
    assert torch.equal(local_d[:, 0].cpu(), expected & (i[:, None] - i[None, :] < 3))


# ----------------------------------------------------------------------------- 3. no leakage

def test_no_leakage():
    g = torch.Generator().manual_seed(1)
    model = tiny_model()
    doc_a, doc_a2, doc_b = rand_doc(100, g), rand_doc(100, g), rand_doc(155, g)
    row1 = torch.cat([doc_a, torch.tensor([EOS]), doc_b])
    row2 = torch.cat([doc_a2, torch.tensor([EOS]), doc_b])                 # only doc A changed
    l1, l2 = logits(model, row1)[0], logits(model, row2)[0]
    diff = (l1[101:] - l2[101:]).abs().max().item()
    assert diff < ATOL, diff
    assert (l1[:101] - l2[:101]).abs().max().item() > 1e-2                  # doc A's own logits do change
    return diff


# ----------------------------------------------------------------------------- 4. single document

def test_single_document_matches_unmasked():
    g = torch.Generator().manual_seed(2)
    masked, plain = tiny_model(doc_mask=True), tiny_model(doc_mask=False)
    plain.load_state_dict(masked.state_dict())
    row = rand_doc(256, g)
    diff = (logits(masked, row) - logits(plain, row)).abs().max().item()
    assert diff < ATOL, diff
    return diff


# ----------------------------------------------------------------------------- 5. packed == alone

def test_packed_equals_alone():
    g = torch.Generator().manual_seed(3)
    model = tiny_model()
    doc_a, doc_b = rand_doc(90, g), rand_doc(166, g)
    packed = logits(model, torch.cat([doc_a, torch.tensor([EOS]), doc_b]))[0, 91:]
    alone = logits(model, doc_b)[0]
    diff = (packed - alone).abs().max().item()
    assert diff < ATOL, diff
    return diff


# ----------------------------------------------------------------------------- 6. labels

def test_boundary_labels_and_loss():
    tokens = torch.tensor([[11, 12, 13, EOS, 21, 22, EOS, 31, 32]])
    x, y = tokens[:, :-1], tokens[:, 1:]
    ym = mask_boundary_labels(x, y, EOS)
    assert (ym[x == EOS] == -100).all()                                     # '<eot> -> next doc' ignored
    assert ym.tolist() == [[12, 13, EOS, -100, 22, EOS, -100, 32]]          # <eot> is still a target
    assert ((ym == -100) == (x == EOS)).all()                              # nothing else touched

    torch.manual_seed(0)
    lg = torch.randn(1, 8, 50)
    valid = ym[0] != -100
    manual = torch.nn.functional.cross_entropy(lg[0][valid], ym[0][valid], reduction="sum") / valid.sum()
    assert torch.allclose(lm_loss(lg, ym), manual)                          # mean over valid tokens only


# ----------------------------------------------------------------------------- 7. flex == dense

def test_flex_equals_dense():
    g = torch.Generator().manual_seed(4)
    model = tiny_model()
    rows = torch.stack([
        torch.cat([rand_doc(60, g), torch.tensor([EOS]), rand_doc(195, g)]),
        torch.cat([rand_doc(200, g), torch.tensor([EOS]), rand_doc(30, g), torch.tensor([EOS]), rand_doc(24, g)]),
        rand_doc(256, g),
    ])
    model.doc_attn_impl = "flex"
    flex = logits(model, rows)
    with torch.no_grad():
        flex_compiled = torch.compile(model)(rows.to(DEV))[0]           # the fused kernel training uses
    model.doc_attn_impl = "dense"
    dense = logits(model, rows)
    diff = max((flex - dense).abs().max().item(), (flex_compiled - dense).abs().max().item())
    assert diff < ATOL, diff
    return diff


if __name__ == "__main__":
    tests = [test_doc_structure, test_mask_block_diagonal, test_no_leakage, test_single_document_matches_unmasked,
             test_packed_equals_alone, test_boundary_labels_and_loss, test_flex_equals_dense]
    failed = 0
    for i, test in enumerate(tests, 1):
        try:
            out = test()
            print(f"PASS {i}. {test.__name__}" + (f"  (max |diff| {out:.2e})" if out is not None else ""))
        except AssertionError as e:
            failed += 1
            print(f"FAIL {i}. {test.__name__}: {e}")
    sys.exit(1 if failed else 0)
