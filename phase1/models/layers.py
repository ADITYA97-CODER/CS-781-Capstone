"""Building blocks shared by the Phase-1 architectures.

  rope_tables / apply_rope   rotary position embeddings (RoPE)
  doc_structure              document id and in-document position of every token in packed rows
  SwiGLU                     gated feed-forward block
  KVCache                    per-layer key/value cache for fast generation
  init_weights               GPT-2-style initialization with depth-scaled residual projections
  GenerationMixin            batched sampling loop (temperature, top-k, top-p, repetition control)

Each architecture file (arch_a_decoder.py, arch_c_hybrid.py, arch_d_sliding.py)
builds its own attention, block and model on top of these.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- RoPE

def rope_tables(head_dim, max_len, theta=10000.0):
    """cos/sin of angle position * theta_i for every position and every pair of dimensions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each (even, odd) pair of dimensions of x: (batch, heads, seq, head_dim).
    After rotation, q . k depends only on the distance between the two positions.
    cos/sin are (seq, head_dim/2) shared by every row, or (batch, seq, head_dim/2) when each
    row has its own positions (doc_mask: positions restart at every document)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = (cos[:, None], sin[:, None]) if cos.dim() == 3 else (cos[None, None], sin[None, None])
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)


def doc_structure(ids, eos_id):
    """ids: (B, L) packed token ids -> (doc_ids, pos_ids), both (B, L).
    A document starts on the token right after each <|endoftext|>; the <|endoftext|> belongs to
    the document it ends. doc_ids counts document starts; pos_ids restart at 0 at every start.
    [A1 A2 A3 <eot> B1 B2 <eot> C1] -> doc_ids [0 0 0 0 1 1 1 2], pos_ids [0 1 2 3 0 1 2 0]."""
    B, L = ids.shape
    new_doc = torch.zeros_like(ids)
    new_doc[:, 1:] = (ids[:, :-1] == eos_id).long()
    doc_ids = torch.cumsum(new_doc, dim=1)
    idx = torch.arange(L, device=ids.device).expand(B, L)
    start = torch.where(new_doc.bool(), idx, torch.zeros_like(idx))
    start = torch.cummax(start, dim=1).values
    return doc_ids, idx - start


# ----------------------------------------------------------------------------- feed-forward

class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): the gate decides, per dimension, how much of up(x) passes."""

    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ----------------------------------------------------------------------------- KV cache

class KVCache:
    """Keys/values of all tokens processed so far in one layer (all rows share one length)."""

    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reorder(self, rows):
        """Beam search: row i now continues the hypothesis that was in row rows[i]."""
        self.k, self.v = self.k[rows], self.v[rows]


# ----------------------------------------------------------------------------- init

def init_weights(model, n_layers):
    """Normal(0, 0.02) for all linear/embedding weights; the two projections that write into the
    residual stream (attention o_proj, FFN down) are scaled by 1/sqrt(2 * n_layers) so the
    residual stream does not grow with depth."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    for name, p in model.named_parameters():
        if name.endswith("o_proj.weight") or name.endswith("down.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))


# ----------------------------------------------------------------------------- generation

class GenerationMixin:
    """Autoregressive generation shared by every decoder-style model.

    The model class provides:
      self.embed, self.blocks, self.norm, self.lm_head, self.rope_cos, self.rope_sin, self.cfg
      and blocks callable as block(x, cos, sin, kv_cache=cache) in causal mode.
    """

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
                 repetition_penalty=1.0, no_repeat_ngram=0, stop_fn=None, ban_eos=False):
        """idx: (B, T) prompts of equal length. temperature=0 -> greedy.
        stop_fn(generated_ids) -> bool (B,) marks rows that are finished.
        ban_eos: never emit <|endoftext|> (its logit is set to -inf).
        A prompt is one document, so generation is plain causal attention even for doc_mask models."""
        B, T = idx.shape
        assert T + max_new_tokens <= self.cfg.max_seq_len, "prompt + generation exceeds max_seq_len"
        caches = [KVCache() for _ in self.blocks]
        out, step_input, pos = idx, idx, 0
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            L = step_input.size(1)               # whole prompt on the first step, then 1 token
            x = self.embed(step_input)
            cos, sin = self.rope_cos[pos:pos + L], self.rope_sin[pos:pos + L]
            for block, cache in zip(self.blocks, caches):
                x = block(x, cos, sin, kv_cache=cache)
            logits = self.lm_head(self.norm(x[:, -1]))[..., :self.cfg.vocab_size].float()   # drop vocab padding
            pos += L
            if ban_eos:
                logits[:, getattr(self.cfg, "eos_id", 0)] = float("-inf")

            if repetition_penalty != 1.0:
                prev = torch.gather(logits, 1, out)
                logits.scatter_(1, out, torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty))
            if no_repeat_ngram > 0 and out.size(1) >= no_repeat_ngram:
                logits = block_repeated_ngrams(logits, out, no_repeat_ngram)
            next_tok = sample(logits, temperature, top_k, top_p)
            next_tok = torch.where(finished, torch.zeros_like(next_tok), next_tok)
            out = torch.cat([out, next_tok[:, None]], dim=1)
            step_input = next_tok[:, None]
            if stop_fn is not None:
                finished |= stop_fn(out[:, T:])
                if finished.all():
                    break
        return out[:, T:]

    @torch.no_grad()
    def beam_generate(self, idx, max_new_tokens, num_beams=4, repetition_penalty=1.0, no_repeat_ngram=0,
                      stop_fn=None, ban_eos=False):
        """Beam search. idx: (B, T) prompts of equal length; returns (B, n) best continuations.
        Every hypothesis has the same token length (no end-of-text token can be emitted), so beams
        are ranked by plain summed log-probability. Stops once every beam satisfies stop_fn; the
        answer is the best-scoring beam that satisfies it (or the best beam if none does)."""
        B, T = idx.shape
        K = num_beams
        assert T + max_new_tokens <= self.cfg.max_seq_len, "prompt + generation exceeds max_seq_len"
        caches = [KVCache() for _ in self.blocks]
        out = idx.repeat_interleave(K, dim=0)                            # (B*K, T), row = b*K + beam
        step_input, pos = out, 0
        scores = torch.full((B, K), float("-inf"), device=idx.device)
        scores[:, 0] = 0.0                                               # identical beams: expand one
        base = (torch.arange(B, device=idx.device) * K)[:, None]
        for _ in range(max_new_tokens):
            L = step_input.size(1)
            x = self.embed(step_input)
            cos, sin = self.rope_cos[pos:pos + L], self.rope_sin[pos:pos + L]
            for block, cache in zip(self.blocks, caches):
                x = block(x, cos, sin, kv_cache=cache)
            logits = self.lm_head(self.norm(x[:, -1]))[..., :self.cfg.vocab_size].float()   # drop vocab padding
            pos += L
            if ban_eos:
                logits[:, getattr(self.cfg, "eos_id", 0)] = float("-inf")

            if repetition_penalty != 1.0:
                prev = torch.gather(logits, 1, out)
                logits.scatter_(1, out, torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty))
            if no_repeat_ngram > 0 and out.size(1) >= no_repeat_ngram:
                logits = block_repeated_ngrams(logits, out, no_repeat_ngram)
            V = logits.size(-1)
            cand = scores[:, :, None] + logits.log_softmax(-1).view(B, K, V)
            scores, flat = cand.view(B, K * V).topk(K, dim=-1)           # best K of all K*V extensions
            rows = (base + flat // V).view(-1)                           # which old beam each came from
            next_tok = (flat % V).view(-1)
            for cache in caches:
                cache.reorder(rows)
            out = torch.cat([out[rows], next_tok[:, None]], dim=1)
            step_input = next_tok[:, None]
            if stop_fn is not None and stop_fn(out[:, T:]).all():
                break

        gen = out[:, T:].view(B, K, -1)
        done = stop_fn(out[:, T:]).view(B, K) if stop_fn is not None else torch.ones_like(scores, dtype=torch.bool)
        best = torch.where(done, scores, scores - 1e9).argmax(-1)       # prefer finished beams
        return gen[torch.arange(B, device=idx.device), best]


def sample(logits, temperature, top_k, top_p):
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p          # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(1)


def block_repeated_ngrams(logits, seq, n):
    """Forbid any token that would complete an n-gram already present in seq (per row), on the GPU."""
    if n == 1:
        return logits.scatter(1, seq, float("-inf"))
    windows = seq.unfold(1, n - 1, 1)[:, :-1]                     # (R, L-n+1, n-1): every (n-1)-gram with a successor
    match = (windows == seq[:, -(n - 1):][:, None, :]).all(-1)   # which of them equal the current suffix
    r, i = match.nonzero(as_tuple=True)
    logits[r, seq[r, i + n - 1]] = float("-inf")                 # ban the token that followed each match
    return logits


def lm_loss(logits, targets, ignore_index=-100):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=ignore_index)
