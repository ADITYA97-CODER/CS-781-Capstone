"""Decoding-time help for copying prompt specifics: prompt-token boost and context-aware decoding (CAD).

Used by generate.py only when --prompt-boost or --cad is set; with both off generate.py calls
model.generate exactly as before.

  PromptBoost    per-row logits processor.
                 start boost:        +b on every fact token id of the row's prompt (prompt_facts.py)
                 continuation boost: if the last n generated tokens equal the first n tokens of a
                                     specific (longest n over all specifics, n >= 1), +b_cont on the
                                     next token of every specific matching with that longest n
  cad_combine    out = (1 + alpha) * logits_full - alpha * logits_short, in fp32, written as
                 logits_full + alpha * (logits_full - logits_short) so that alpha = 0 or
                 logits_short == logits_full returns logits_full bit-exactly; tokens with
                 p_full < beta * max p_full get -inf (plausibility filter).
  guided_generate the GenerationMixin.generate loop plus a second KV cache for CAD: the short run
                 sees only the last `short_len` prompt tokens (RoPE positions from 0, as a prompt of
                 that length) followed by the same generated tokens. Both runs go through the
                 model's own blocks and KVCache, so local layers keep their sliding window.
                 Per step: CAD -> prompt boost -> repetition penalty / no-repeat -> sampling
                 (temperature, top-k, top-p), the last two unchanged from GenerationMixin.
  cad adaptive   cad["js_ref"] set: per row and step, alpha_t = alpha * clamp(JS(p_full, p_short) / js_ref, 0, 1)
                 (JS in nats). js_ref = inf -> alpha_t = 0 (normal decoding); tiny js_ref -> fixed alpha.
                 cad["record_js"] (a list) collects the JS of unfinished rows (calibration).
  no_repeat_gen_ngram  ban any token that would repeat an n-gram already present in the GENERATED text
                 (n-grams of the prompt, or copied from it once, stay allowed).
  steer_*        negative PromptBoosts (generate.py --steer-headers / --steer-close / --steer-cite):
                 penalize the next token of a court-header or closing-formula phrase once its prefix
                 has been generated, single-token header markers, and citation reporters (SCC, AIR, ...)
                 that do not occur in the row's prompt. `boost` may be a list of processors.
"""

import torch

from models.layers import KVCache, block_repeated_ngrams, sample


class PromptBoost:
    def __init__(self, seqs, ids, vocab_size, b=1.0, b_cont=4.0, device="cuda"):
        """seqs: per row, list of token-id sequences (one per specific); ids: per row, fact token ids."""
        B = len(seqs)
        self.b, self.b_cont = b, b_cont
        self.fact_mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
        for r, row in enumerate(ids):
            if row:
                self.fact_mask[r, torch.tensor(row, device=device)] = True
        N = max([len(s) for s in seqs] + [1])
        L = max([len(x) for s in seqs for x in s] + [2])
        self.S = torch.full((B, N, L), -1, dtype=torch.long, device=device)   # padded specifics
        self.len = torch.zeros(B, N, dtype=torch.long, device=device)
        for r, row in enumerate(seqs):
            for j, x in enumerate(row):
                self.S[r, j, :len(x)] = torch.tensor(x, device=device)
                self.len[r, j] = len(x)
        # match lengths n = 1..L-1; for length n, prefix position i < n is compared with suffix position
        # K - n + i of the last K = L-1 generated tokens
        K = L - 1
        n = torch.arange(1, K + 1, device=device)[:, None]
        i = torch.arange(K, device=device)[None, :]
        self.valid = i < n                                              # (K, K): [n-1, i]
        self.gather_at = torch.where(self.valid, K - n + i, 0)          # (K, K)
        self.n = n.squeeze(1)                                           # (K,)

    def continuation_targets(self, gen):
        """gen: (B, t) generated ids -> (B, V) bool mask of tokens that continue the longest partial match.
        All match lengths are tested at once (no host syncs)."""
        B, t = gen.shape
        K = self.n.numel()
        tail = torch.full((B, K), -2, dtype=gen.dtype, device=gen.device)  # -2 never equals a token
        tail[:, K - min(K, t):] = gen[:, t - min(K, t):]
        suffix = tail[:, self.gather_at]                                # (B, n, i)
        eq = (self.S[:, :, None, :K] == suffix[:, None]) | ~self.valid  # (B, N, n, i)
        m = eq.all(-1) & (self.len[:, :, None] > self.n) & (self.n <= t)   # (B, N, n): prefix of length n matches
        best = (m.any(1) * self.n).max(-1).values                       # longest match per row (0 = none)
        sel = m.gather(2, (best - 1).clamp(min=0)[:, None, None].expand(-1, m.size(1), 1)).squeeze(2)
        sel &= (best > 0)[:, None]                                      # (B, N) specifics matching with n = best
        nxt = self.S.gather(2, best[:, None, None].expand(-1, m.size(1), 1)).squeeze(2).clamp(min=0)
        hit = torch.zeros(B, self.fact_mask.size(1), device=gen.device).scatter_add_(1, nxt, sel.float())
        return hit > 0

    def __call__(self, logits, gen):
        if self.b:
            logits = logits + self.b * self.fact_mask
        if self.b_cont and gen.size(1) > 0:
            logits = logits + self.b_cont * self.continuation_targets(gen)
        return logits


def cad_combine(logits_full, logits_short, alpha, beta):
    lf, ls = logits_full.float(), logits_short.float()
    finite = torch.isfinite(lf) & torch.isfinite(ls)                 # banned specials are -inf in both
    out = lf + alpha * torch.where(finite, lf - ls, torch.zeros_like(lf))
    if beta > 0:
        p = lf.softmax(-1)
        out = out.masked_fill(p < beta * p.max(-1, keepdim=True).values, float("-inf"))
    return out


def js_divergence(logits_a, logits_b):
    """Jensen-Shannon divergence (nats) between softmax(logits_a) and softmax(logits_b), per row, fp32."""
    p, q = logits_a.float().softmax(-1), logits_b.float().softmax(-1)
    m = 0.5 * (p + q)
    kl = lambda x: torch.where(x > 0, x * (torch.log(x.clamp_min(1e-30)) - torch.log(m.clamp_min(1e-30))),
                               torch.zeros_like(x)).sum(-1)
    return 0.5 * kl(p) + 0.5 * kl(q)


def ban_generated_repeats(logits, gen, n):
    """-inf on any token that would complete an n-gram already present in the generated tokens `gen`
    (prompt n-grams are not looked at, so copying one from the prompt once stays possible)."""
    return block_repeated_ngrams(logits, gen, n) if gen.size(1) >= n else logits


def forward_step(model, caches, step_input, pos):
    """One incremental step through the model's own blocks and caches (as GenerationMixin.generate)."""
    L = step_input.size(1)
    x = model.embed(step_input)
    cos, sin = model.rope_cos[pos:pos + L], model.rope_sin[pos:pos + L]
    for block, cache in zip(model.blocks, caches):
        x = block(x, cos, sin, kv_cache=cache)
    return model.lm_head(model.norm(x[:, -1]))[..., :model.cfg.vocab_size].float()


@torch.no_grad()
def guided_generate(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                    no_repeat_ngram=0, stop_fn=None, boost=None, cad=None, return_logits=False,
                    no_repeat_gen_ngram=0, stats=None):
    """idx: (B, T) prompts of equal length. boost: PromptBoost or None.
    cad: dict(alpha=, short_len=, beta=[, js_ref=, record_js=]) or None. return_logits: also return per-step
    (logits_full, logits_short) for tests."""
    B, T = idx.shape
    assert T + max_new_tokens <= model.cfg.max_seq_len, "prompt + generation exceeds max_seq_len"
    caches = [KVCache() for _ in model.blocks]
    out, step_input, pos = idx, idx, 0
    if cad is not None:
        short_caches = [KVCache() for _ in model.blocks]
        short_input, short_pos = idx[:, -cad["short_len"]:], 0
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    trace = []
    for _ in range(max_new_tokens):
        logits = forward_step(model, caches, step_input, pos)
        raw = logits.clone() if no_repeat_gen_ngram > 0 else None        # fallback for fully banned rows
        pos += step_input.size(1)
        if cad is not None:
            logits_short = forward_step(model, short_caches, short_input, short_pos)
            short_pos += short_input.size(1)
            if return_logits:
                trace.append((logits, logits_short))
            alpha = cad["alpha"]
            if cad.get("js_ref") is not None or cad.get("record_js") is not None:
                js = js_divergence(logits, logits_short)
                if cad.get("record_js") is not None:
                    cad["record_js"].extend(js[~finished].tolist())
                if cad.get("js_ref") is not None:
                    alpha = cad["alpha"] * (js / cad["js_ref"]).clamp(0, 1)[:, None]
            logits = cad_combine(logits, logits_short, alpha, cad["beta"])
        for proc in (boost if isinstance(boost, (list, tuple)) else [boost] if boost is not None else []):
            logits = proc(logits, out[:, T:])

        if repetition_penalty != 1.0:
            prev = torch.gather(logits, 1, out)
            logits.scatter_(1, out, torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty))
        if no_repeat_ngram > 0 and out.size(1) >= no_repeat_ngram:
            logits = block_repeated_ngrams(logits, out, no_repeat_ngram)
        if no_repeat_gen_ngram > 0:
            logits = ban_generated_repeats(logits, out[:, T:], no_repeat_gen_ngram)
            # CAD's plausibility filter can leave only tokens that the ban removes; such a row falls back to the
            # plain full-context logits (no CAD / filter) with the same ban
            dead = ~torch.isfinite(logits).any(-1, keepdim=True)
            logits = torch.where(dead, ban_generated_repeats(raw, out[:, T:], no_repeat_gen_ngram), logits)
            if stats is not None:
                stats["fallback_steps"] = stats.get("fallback_steps", 0) + dead.sum()
        next_tok = sample(logits, temperature, top_k, top_p)
        next_tok = torch.where(finished, torch.zeros_like(next_tok), next_tok)
        out = torch.cat([out, next_tok[:, None]], dim=1)
        step_input = short_input = next_tok[:, None]
        if stop_fn is not None:
            finished |= stop_fn(out[:, T:])
            if finished.all():
                break
    return (out[:, T:], trace) if return_logits else out[:, T:]


HEADER_PHRASES = ["IN THE SUPREME COURT", "IN THE HIGH COURT", "SUPREME COURT OF INDIA", "Whether reporters of the local papers",
                  "Whether the reporters of the local papers", "ITEM NO", "S U P R E M E"]
HEADER_TOKENS = ["REPORTABLE", "JURISDICTION", "JUDGE"]
CLOSE_PHRASES = ["appeal is dismissed", "appeal is allowed", "appeals are dismissed", "appeals are allowed",
                 "petition is dismissed", "petition is allowed", "petition is disposed of", "petitions are dismissed",
                 "No order as to costs", "no order as to costs", "No orders as to costs", "no orders as to costs",
                 "stands disposed of", "stand disposed of", "parties to bear", "Rule is made absolute",
                 "Rule made absolute", "Rule is discharged", "bail bonds"]
REPORTERS = ["SCC", "SCR", "AIR", "Cri", "SCALE", "JT", "Supp"]


def _variants(tokenizer, phrase):
    """token-id sequences of a phrase with and without a leading space."""
    return [tokenizer(v, add_special_tokens=False)["input_ids"] for v in (" " + phrase, phrase)]


def steer_phrases(tokenizer, B, vocab_size, phrases, tokens=(), penalty=5.0, device="cuda"):
    """Same for every row: -penalty on the next token of any phrase whose prefix was just generated, and on
    single marker tokens."""
    seqs = [s for p in phrases for s in _variants(tokenizer, p) if len(s) > 1]
    ids = sorted({s[0] for t in tokens for s in _variants(tokenizer, t) if len(s) == 1})
    return PromptBoost([seqs] * B, [ids] * B, vocab_size, b=-penalty, b_cont=-penalty, device=device)


def steer_citations(tokenizer, prompt_texts, vocab_size, penalty=3.0, device="cuda"):
    """Per row: -penalty on citation reporter tokens that do not occur in that row's prompt."""
    rep_ids = {r: {s[0] for s in _variants(tokenizer, r) if len(s) == 1} for r in REPORTERS}
    ids = [sorted({i for r, v in rep_ids.items() if r not in t for i in v}) for t in prompt_texts]
    return PromptBoost([[] for _ in prompt_texts], ids, vocab_size, b=-penalty, b_cont=0.0, device=device)
