"""Phase 6 model code: a Phase-1 decoder (Arch E / F) as a decoder-only summarizer, adapters, and batched generation.

Training runs the Phase-1 blocks on packed rows with document masking (FlexAttention BlockMasks from the
<|endoftext|> structure, as in pretraining); logits are computed only at labelled (summary) positions.

Generation takes prompts of different lengths: they are left-padded, padding keys are masked out and RoPE positions
start at 0 at the first real token of every row, so a row sees exactly what it saw in training. Dense boolean masks +
SDPA with a KV cache. Modes: sampling / greedy (beams 1) and beam search (finished hypotheses ranked by
log-probability / length^len_pen). Special ids 1-104 are never emitted; <|endoftext|> is banned before min_new tokens.
Repetition penalty and n-gram blocking look at the generated tokens only (copying from the document stays allowed).
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase1"))
from models import build_model  # noqa: E402
from models.layers import block_repeated_ngrams, doc_structure, rope_tables, sample  # noqa: E402

EOS = 0
N_SPECIAL = 105


def load_lm(ckpt_path, max_len=8192):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, cfg = build_model(state["config"]["model"])
    model.load_state_dict(state["model"])
    if max_len > model.rope_cos.size(0):          # longer than pretraining (1536)
        model.rope_cos, model.rope_sin = rope_tables(cfg.head_dim, max_len, cfg.rope_theta)
    cfg.max_seq_len = max(cfg.max_seq_len, max_len)
    return model, cfg


# ----------------------------------------------------------------------------- adapters

class BottleneckAdapter(nn.Module):
    """y + up(gelu(down(y))) on a linear's output (Houlsby et al. 2019); up starts at zero (identity at init)."""

    def __init__(self, base: nn.Linear, size=64):
        super().__init__()
        self.base = base
        d = base.out_features
        self.down = nn.Linear(d, size)
        self.up = nn.Linear(size, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))


def add_bottleneck_adapters(backbone, kind="houlsby", size=64):
    """houlsby: an adapter after the attention output projection and after the FFN (2 per block);
    pfeiffer: after the FFN only. Freezes the backbone except the RMSNorm gains (trained as in both papers' setups)."""
    for p in backbone.parameters():
        p.requires_grad = False
    n = 0
    for block in backbone.blocks:
        if kind == "houlsby":
            block.attn.o_proj = BottleneckAdapter(block.attn.o_proj, size)
            n += 1
        block.ffn.down = BottleneckAdapter(block.ffn.down, size)
        n += 1
    for m in backbone.modules():
        if isinstance(m, nn.RMSNorm):
            m.weight.requires_grad = True
    return n


# ----------------------------------------------------------------------------- training forward

class Decoder(nn.Module):
    """Phase-1 blocks over packed rows -> final-norm hidden states (B, T, d). Compiled separately (static shapes)."""

    def __init__(self, backbone, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        for block in backbone.blocks:
            block.drop.p = dropout

    def forward(self, ids, pos, masks):
        bb = self.backbone
        x = bb.embed(ids)
        cos, sin = bb.rope_cos[pos], bb.rope_sin[pos]
        for block in bb.blocks:
            x = block(x, cos, sin, masks[block.attn.local])
        return bb.norm(x)


class Summarizer(nn.Module):
    def __init__(self, backbone, dropout=0.0):
        super().__init__()
        self.decoder = Decoder(backbone, dropout)
        self.compiled = None

    def set_compiled(self, fn):
        object.__setattr__(self, "compiled", fn)

    @property
    def backbone(self):
        return self.decoder.backbone

    def hidden(self, ids):
        bb = self.backbone
        doc_ids, pos = doc_structure(ids, EOS)
        masks = bb.doc_masks(doc_ids)
        return (self.decoder if self.compiled is None else self.compiled)(ids, pos, masks)

    def logits_at(self, h, sel):
        """h (B, T, d), sel (B, T) bool -> (n, vocab) fp32 logits of the selected positions."""
        bb = self.backbone
        return bb.lm_head(h[sel])[..., :bb.cfg.vocab_size].float()

    def forward(self, ids, lab):
        """-> (sum of token CE over labelled positions, number of labelled positions, logits, selection)."""
        h = self.hidden(ids)
        sel = lab != -100
        logits = self.logits_at(h, sel)
        y = lab[sel]
        return F.cross_entropy(logits, y, reduction="sum"), int(y.numel()), logits, sel


# ----------------------------------------------------------------------------- generation

def _layer_step(block, x, cos, sin, cache, mask):
    a = block.attn
    q, k, v = a.project(block.attn_norm(x), cos, sin)
    if cache[0] is not None:
        k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
    cache[0], cache[1] = k, v
    x = x + a.o_proj(a.combine(q, k, v, mask))
    return x + block.ffn(block.ffn_norm(x))


@torch.no_grad()
def prefill(bb, prompts, device, rows=4):
    """Left-pad prompts (list of 1-D LongTensors) -> (caches [[k, v]] per layer, last hidden (B, d), pad (B,), T)."""
    B, T = len(prompts), max(len(p) for p in prompts)
    pad = torch.tensor([T - len(p) for p in prompts], device=device)
    ids = torch.full((B, T), EOS, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, T - len(p):] = p.to(device)
    j = torch.arange(T, device=device)
    valid = j[None] >= pad[:, None]                                   # (B, T) real tokens
    pos = (j[None] - pad[:, None]).clamp(min=0)
    causal = j[:, None] >= j[None, :]
    window = bb.cfg.window
    caches = [[[], []] for _ in bb.blocks]
    last = []
    for s in range(0, B, rows):
        sl = slice(s, s + rows)
        glob = (causal[None] & valid[sl, None, :]) | torch.eye(T, dtype=torch.bool, device=device)[None]
        masks = {False: glob[:, None], True: (glob & (j[:, None] - j[None, :] < window))[:, None]}
        x = bb.embed(ids[sl])
        cos, sin = bb.rope_cos[pos[sl]], bb.rope_sin[pos[sl]]
        for block, c in zip(bb.blocks, caches):
            cache = [None, None]
            x = _layer_step(block, x, cos, sin, cache, masks[block.attn.local])
            c[0].append(cache[0])
            c[1].append(cache[1])
        last.append(bb.norm(x[:, -1]))
    caches = [[torch.cat(c[0]), torch.cat(c[1])] for c in caches]
    return caches, torch.cat(last), pad, T


def _logits(bb, h):
    return bb.lm_head(h)[..., :bb.cfg.vocab_size].float()


def _process(logits, gen, t, args):
    logits[:, 1:N_SPECIAL] = float("-inf")
    if t < args.min_new:
        logits[:, EOS] = float("-inf")
    if args.rep_pen != 1.0 and gen.size(1):
        prev = torch.gather(logits, 1, gen)
        logits.scatter_(1, gen, torch.where(prev > 0, prev / args.rep_pen, prev * args.rep_pen))
    if args.no_repeat and gen.size(1) >= args.no_repeat:
        logits = block_repeated_ngrams(logits, gen, args.no_repeat)
    return logits


def _decode_step(bb, tok, caches, pos, pad, Tk):
    """One new token per row. tok (R,), pos (R,) its position, pad (R,), Tk = keys after this step."""
    device = tok.device
    j = torch.arange(Tk, device=device)
    valid = (j[None] >= pad[:, None])[:, None, None, :]               # (R, 1, 1, Tk)
    local = valid & (Tk - 1 - j < bb.cfg.window)[None, None, None, :]
    x = bb.embed(tok[:, None])
    cos, sin = bb.rope_cos[pos[:, None]], bb.rope_sin[pos[:, None]]
    for block, cache in zip(bb.blocks, caches):
        x = _layer_step(block, x, cos, sin, cache, local if block.attn.local else valid)
    return _logits(bb, bb.norm(x[:, -1]))


@torch.no_grad()
def generate(bb, prompts, args, device):
    """prompts: list of 1-D LongTensors -> list of generated token lists (without the end token).
    args: max_new, min_new, beams, temperature (0 = greedy), top_p, rep_pen, no_repeat, len_pen."""
    caches, h, pad, T = prefill(bb, prompts, device)
    B = len(prompts)
    plen = torch.tensor([len(p) for p in prompts], device=device)
    if args.beams <= 1:
        gen = torch.zeros(B, 0, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        logits = _logits(bb, h)
        for t in range(args.max_new):
            nxt = sample(_process(logits, gen, t, args), args.temperature, 0, args.top_p)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            gen = torch.cat([gen, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
            logits = _decode_step(bb, nxt, caches, plen + t, pad, T + t + 1)
        out = []
        for row in gen.tolist():
            out.append(row[:row.index(EOS)] if EOS in row else row)
        return out

    K = args.beams
    caches = [[c[0].repeat_interleave(K, 0), c[1].repeat_interleave(K, 0)] for c in caches]
    pad, plen = pad.repeat_interleave(K), plen.repeat_interleave(K)
    logits = _logits(bb, h).repeat_interleave(K, 0)
    scores = torch.full((B, K), float("-inf"), device=device)
    scores[:, 0] = 0.0
    gen = torch.zeros(B * K, 0, dtype=torch.long, device=device)
    base = (torch.arange(B, device=device) * K)[:, None]
    finished = [[] for _ in range(B)]                                 # (normalised score, tokens)
    for t in range(args.max_new):
        logp = _process(logits, gen, t, args).log_softmax(-1)
        V = logp.size(-1)
        cand = (scores[:, :, None] + logp.view(B, K, V)).view(B, K * V)
        vals, flat = cand.topk(2 * K, dim=-1)                         # (B, 2K), sorted
        is_eos = (flat % V) == EOS
        rank = torch.arange(2 * K, device=device)[None]
        hit = (is_eos & (rank < K) & torch.isfinite(vals)).nonzero().tolist()
        for b, r in hit:
            if len(finished[b]) < K:
                row = int(base[b, 0] + flat[b, r] // V)
                finished[b].append((vals[b, r].item() / max(1, t) ** args.len_pen, gen[row].tolist()))
        vals = vals.masked_fill(is_eos, float("-inf"))
        top = vals.topk(K, dim=-1)
        scores = top.values
        pick = flat.gather(1, top.indices)
        rows = (base + pick // V).view(-1)
        nxt = (pick % V).view(-1)
        for c in caches:
            c[0], c[1] = c[0][rows], c[1][rows]
        gen = torch.cat([gen[rows], nxt[:, None]], 1)
        if all(len(f) >= K for f in finished):
            break
        logits = _decode_step(bb, nxt, caches, plen + t, pad, T + t + 1)
    out = []
    sc = scores.tolist()
    for b in range(B):
        f = finished[b]
        if len(f) < K:                                                # max_new reached: live beams count too
            f = f + [(sc[b][k] / max(1, gen.size(1)) ** args.len_pen, gen[b * K + k].tolist()) for k in range(K)
                     if math.isfinite(sc[b][k])]
        out.append(max(f, key=lambda z: z[0])[1])
    return out
