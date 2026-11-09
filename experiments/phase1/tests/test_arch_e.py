"""Tests for Arch E (hybrid local/global + differential attention + tied embeddings).

Run on GPU 0 (FlexAttention needs CUDA):
  CUDA_VISIBLE_DEVICES=0 python phase1/tests/test_arch_e.py
Tests 3-9 use tiny random models in fp32; tests 1, 2 and 10 use the real config (configs/archE.json).
Test 11 (throughput) is phase1/tests/bench_arch.py.
"""

import io
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import mask_boundary_labels  # noqa: E402
from models import build_model  # noqa: E402
from models.arch_e_diff import ArchEConfig, ArchEDiff  # noqa: E402
from models.layers import KVCache, doc_structure, lm_loss, rope_tables  # noqa: E402
from train import make_optimizer  # noqa: E402

EOS = 0
DEV = "cuda"
ATOL = 1e-5
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCH_E_CFG = json.load(open(os.path.join(PROJECT_DIR, "phase1", "configs", "archE.json")))["model"]


def tiny_cfg(**kw):
    base = dict(vocab_size=200, vocab_pad_to=128, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
                ffn_hidden=128, max_seq_len=512, window=16, layer_pattern="alternate", diff_layers="all",
                doc_mask=True, eos_id=EOS)
    base.update(kw)
    return ArchEConfig(**base)


def tiny_model(**kw):
    torch.manual_seed(0)
    return ArchEDiff(tiny_cfg(**kw)).to(DEV).eval()


def rand_doc(n, gen):
    return torch.randint(5, 200, (n,), generator=gen)            # never <|endoftext|>


def logits(model, rows):
    rows = rows[None] if rows.dim() == 1 else rows
    with torch.no_grad():
        return model(rows.to(DEV))[0]


def component_counts(model):
    """Parameters per component; model.parameters() yields a tied matrix once."""
    seen, out = set(), {"embedding": 0, "attention": 0, "mlp": 0, "norms": 0, "lambda": 0}
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith(("embed", "lm_head")):
            k = "embedding"
        elif ".attn.lambda_" in name or name.endswith(".attn.subln"):
            k = "lambda"
        elif ".attn." in name:                                    # q/k/v/o projections + QK-norm
            k = "attention"
        elif ".ffn." in name:
            k = "mlp"
        else:
            k = "norms"
        out[k] += p.numel()
    out["total"] = sum(out.values())
    out["non_embedding"] = out["total"] - out["embedding"]
    return out


# ----------------------------------------------------------------------------- 1. parameter count

def test_param_count():
    with torch.device("meta"):
        model, cfg = build_model(ARCH_E_CFG)
    c = component_counts(model)
    assert c["total"] == sum(p.numel() for p in model.parameters())
    assert c["embedding"] == cfg.padded_vocab * cfg.d_model                  # tied matrix counted once
    assert c["total"] == 144_561_408, c["total"]
    assert c["total"] < 150_000_000
    return ", ".join(f"{k} {v:,}" for k, v in c.items())


# ----------------------------------------------------------------------------- 2. diff overhead

def test_diff_param_overhead():
    with torch.device("meta"):
        diff, cfg = build_model(ARCH_E_CFG)
        plain, _ = build_model({**ARCH_E_CFG, "diff_layers": "none"})
        glob, _ = build_model({**ARCH_E_CFG, "diff_layers": "global_only"})
    n = lambda m: sum(p.numel() for p in m.parameters())
    per_layer = 4 * cfg.head_dim + cfg.d_model
    n_global = sum(not cfg.is_local(i) for i in range(cfg.n_layers))
    assert n(diff) == n(plain) + cfg.n_layers * per_layer, (n(diff), n(plain))
    assert n(glob) == n(plain) + n_global * per_layer
    return f"standard {n(plain):,} + {cfg.n_layers} x {per_layer} = {n(diff):,}"


# ----------------------------------------------------------------------------- 3. diff attention vs naive

def naive_diff_attention(attn, x, cos, sin, allowed):
    """Explicit softmax matrices: A1 = softmax(q1 k1^T) V, A2 = softmax(q2 k2^T) V, A1 - lam A2, ...
    allowed: (B, T, T) boolean."""
    q, k, v = attn.project(x, cos, sin)                             # (B, 2H, T, hd), (B, 2K, T, hd), (B, K, T, 2hd)
    H, K = attn.n_heads, attn.n_kv
    rep = H // K
    lam = attn.lam()
    outs = []
    for h in range(H):
        g = h // rep                                                # GQA: query head h uses KV head g
        maps = []
        for c in range(2):
            s = q[:, 2 * h + c] @ k[:, 2 * g + c].transpose(-1, -2) / math.sqrt(attn.hd)
            s = s.masked_fill(~allowed, float("-inf"))
            maps.append(torch.softmax(s, dim=-1))
        o = maps[0] @ v[:, g] - lam * (maps[1] @ v[:, g])           # (B, T, 2hd)
        o = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + attn.norm_eps) * attn.subln[h]
        outs.append(o * (1 - attn.lambda_init))
    return attn.o_proj(torch.cat(outs, dim=-1))


def test_diff_matches_naive():
    g = torch.Generator().manual_seed(5)
    model = tiny_model(window=48)
    rows = torch.stack([torch.cat([rand_doc(70, g), torch.tensor([EOS]), rand_doc(129, g)]),
                        rand_doc(200, g)]).to(DEV)
    doc_ids, pos_ids = doc_structure(rows, EOS)
    cos, sin = model.rope_cos[pos_ids], model.rope_sin[pos_ids]
    masks = model.doc_masks(doc_ids)
    torch.manual_seed(1)
    x = torch.randn(2, 200, 64, device=DEV)
    T = rows.size(1)
    i = torch.arange(T, device=DEV)
    worst = 0.0
    for layer in (0, 1):                                            # layer 0 local, layer 1 global
        attn = model.blocks[layer].attn
        assert attn.diff and attn.local == (layer == 0)
        allowed = (doc_ids[:, :, None] == doc_ids[:, None, :]) & (i[:, None] >= i[None, :])
        if attn.local:
            allowed &= (i[:, None] - i[None, :] < attn.window)
        with torch.no_grad():
            got = attn(x, cos, sin, masks[attn.local])
            ref = naive_diff_attention(attn, x, cos, sin, allowed)
        worst = max(worst, (got - ref).abs().max().item())
    assert worst < ATOL, worst
    return worst


# ----------------------------------------------------------------------------- 4. causality

def test_causality():
    g = torch.Generator().manual_seed(6)
    model = tiny_model()
    row = rand_doc(120, g)
    t = 70
    row2 = row.clone()
    row2[t] = (row[t] + 1 - 5) % 195 + 5                            # different non-EOS token
    l1, l2 = logits(model, row)[0], logits(model, row2)[0]
    before = (l1[:t] - l2[:t]).abs().max().item()
    assert before < ATOL, before
    assert (l1[t:] - l2[t:]).abs().max().item() > 1e-3
    return before


# ----------------------------------------------------------------------------- 5. doc mask

def test_doc_mask_packed():
    g = torch.Generator().manual_seed(7)
    model = tiny_model()                                            # window 16 < doc lengths: local + global
    doc_a, doc_a2, doc_b = rand_doc(90, g), rand_doc(90, g), rand_doc(166, g)
    row1 = torch.cat([doc_a, torch.tensor([EOS]), doc_b])
    row2 = torch.cat([doc_a2, torch.tensor([EOS]), doc_b])
    l1, l2 = logits(model, row1)[0], logits(model, row2)[0]
    leak = (l1[91:] - l2[91:]).abs().max().item()
    alone = (l1[91:] - logits(model, doc_b)[0]).abs().max().item()
    assert leak < ATOL, leak
    assert alone < ATOL, alone
    return max(leak, alone)


def test_flex_equals_dense():
    g = torch.Generator().manual_seed(8)
    model = tiny_model()
    rows = torch.stack([torch.cat([rand_doc(60, g), torch.tensor([EOS]), rand_doc(195, g)]),
                        torch.cat([rand_doc(200, g), torch.tensor([EOS]), rand_doc(30, g), torch.tensor([EOS]),
                                   rand_doc(24, g)]),
                        rand_doc(256, g)])
    worst = 0.0
    for doc_mask in (True, False):
        model.cfg.doc_mask = doc_mask
        model.doc_attn_impl = "flex"
        flex = logits(model, rows)
        model.doc_attn_impl = "dense"
        dense = logits(model, rows)
        worst = max(worst, (flex - dense).abs().max().item())
    model.cfg.doc_mask, model.doc_attn_impl = True, "flex"
    assert worst < 1e-4, worst
    return worst


# ----------------------------------------------------------------------------- 6. local window

def test_local_window():
    g = torch.Generator().manual_seed(9)
    W = 16
    model = tiny_model(n_layers=1, layer_pattern=["L"], window=W, doc_mask=False)
    row = rand_doc(100, g)
    j = 20
    row2 = row.clone()
    row2[j] = (row[j] + 1 - 5) % 195 + 5
    d = (logits(model, row)[0] - logits(model, row2)[0]).abs().amax(-1)
    assert d[:j].max().item() < ATOL                                # before j: causal
    assert (d[j:j + W] > 1e-4).all()                                # j .. j+W-1 see token j
    assert d[j + W:].max().item() < ATOL, d[j + W:].max()           # >= W back: invisible
    # with doc_mask the same window applies inside a document
    model.cfg.doc_mask = True
    d2 = (logits(model, row)[0] - logits(model, row2)[0]).abs().amax(-1)
    assert d2[j + W:].max().item() < ATOL and (d2[j:j + W] > 1e-4).all()
    return d[j + W:].max().item()


# ----------------------------------------------------------------------------- 7. labels

def test_boundary_labels():
    tokens = torch.tensor([[11, 12, 13, EOS, 21, 22, EOS, 31, 32]])
    x, y = tokens[:, :-1], tokens[:, 1:]
    ym = mask_boundary_labels(x, y, EOS)
    assert ym.tolist() == [[12, 13, EOS, -100, 22, EOS, -100, 32]]          # <eot> stays a target
    assert ((ym == -100) == (x == EOS)).all()
    torch.manual_seed(0)
    lg = torch.randn(1, 8, 50)
    valid = ym[0] != -100
    manual = nn.functional.cross_entropy(lg[0][valid], ym[0][valid], reduction="sum") / valid.sum()
    assert torch.allclose(lm_loss(lg, ym), manual)


# ----------------------------------------------------------------------------- 8. KV cache

def incremental_logits(model, rows, prefill):
    """Same loop as GenerationMixin.generate: prefill, then one token at a time with KV caches."""
    caches = [KVCache() for _ in model.blocks]
    out, pos = [], 0
    with torch.no_grad():
        for s, e in [(0, prefill)] + [(t, t + 1) for t in range(prefill, rows.size(1))]:
            x = model.embed(rows[:, s:e])
            cos, sin = model.rope_cos[pos:pos + e - s], model.rope_sin[pos:pos + e - s]
            for block, cache in zip(model.blocks, caches):
                x = block(x, cos, sin, kv_cache=cache)
            out.append(model.lm_head(model.norm(x))[..., :model.cfg.vocab_size])
            pos += e - s
    return torch.cat(out, dim=1)


def test_kv_cache():
    g = torch.Generator().manual_seed(10)
    model = tiny_model(window=16)                                   # layers L G L G, all differential
    assert [b.attn.local for b in model.blocks] == [True, False, True, False]
    assert all(b.attn.diff for b in model.blocks)
    rows = torch.stack([rand_doc(60, g), rand_doc(60, g)]).to(DEV)  # 60 > window: the window matters
    full = logits(model, rows)
    worst = 0.0
    for prefill in (1, 20):
        worst = max(worst, (incremental_logits(model, rows, prefill) - full).abs().max().item())
    assert worst < 1e-4, worst
    # bf16 generation through GenerationMixin runs (SDPA with value width 2*head_dim)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(rows[:, :20], 30, temperature=0.0)
    assert gen.shape == (2, 30)
    return worst


# ----------------------------------------------------------------------------- 9. compile == eager

def test_compile_matches_eager():
    g = torch.Generator().manual_seed(11)
    model = tiny_model()
    rows = torch.stack([torch.cat([rand_doc(100, g), torch.tensor([EOS]), rand_doc(156, g)]),
                        rand_doc(257, g)]).to(DEV)
    x, y = rows[:, :-1], mask_boundary_labels(rows[:, :-1], rows[:, 1:], EOS)
    with torch.no_grad():
        eager = model(x, y)[1].item()
        compiled = torch.compile(model)(x, y)[1].item()
    diff = abs(eager - compiled)
    assert diff < 1e-4, (eager, compiled)
    return diff


# ----------------------------------------------------------------------------- 10. embedding tying

def test_tied_embeddings():
    model, cfg = build_model(ARCH_E_CFG)
    model = model.to(DEV)
    same = lambda m: m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr() and m.lm_head.weight is m.embed.weight
    assert same(model), "after init"
    std = model.embed.weight.std().item()
    assert abs(std - 0.02) < 1e-3, std

    compiled = torch.compile(model)
    assert same(compiled) and compiled.lm_head.weight.data_ptr() == model.embed.weight.data_ptr(), "after compile"

    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "config": {"model": ARCH_E_CFG}}, buf)
    buf.seek(0)
    state = torch.load(buf, map_location="cpu", weights_only=False)
    loaded, _ = build_model(state["config"]["model"])
    loaded.load_state_dict(state["model"])
    loaded = loaded.to(DEV)
    assert same(loaded), "after save + load"
    assert torch.equal(loaded.embed.weight, model.embed.weight)

    opt = make_optimizer(model, {"wd_linear_only": True, "weight_decay": 0.1, "lr": 1e-3, "beta2": 0.95})
    ids = [id(p) for grp in opt.param_groups for p in grp["params"]]
    assert ids.count(id(model.embed.weight)) == 1, "tied matrix must be in exactly one param group"
    assert len(ids) == len(set(ids))
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(model.embed.weight) not in decay_ids
    linear = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)} - {id(model.embed.weight)}
    assert decay_ids == linear                                      # no decay on norms, lambdas, subln, embedding

    before = model.embed.weight.detach().clone()
    g = torch.Generator().manual_seed(12)
    rows = torch.randint(5, 32000, (2, 129), generator=g).to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = compiled(rows[:, :-1], rows[:, 1:])[1]
    loss.backward()
    opt.step()
    assert same(model), "after optimizer step"
    assert torch.equal(model.lm_head.weight, model.embed.weight)
    assert not torch.equal(model.embed.weight, before), "the shared matrix was updated"
    return f"embedding std {std:.4f}; {len(ids)} tensors in optimizer, shared matrix once (no decay)"


if __name__ == "__main__":
    tests = [test_param_count, test_diff_param_overhead, test_diff_matches_naive, test_causality,
             test_doc_mask_packed, test_flex_equals_dense, test_local_window, test_boundary_labels, test_kv_cache,
             test_compile_matches_eager, test_tied_embeddings]
    failed = 0
    for test in tests:
        try:
            out = test()
            msg = f"  ({out})" if isinstance(out, str) else (f"  (max |diff| {out:.2e})" if out is not None else "")
            print(f"PASS {test.__name__}{msg}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}", flush=True)
    sys.exit(1 if failed else 0)
