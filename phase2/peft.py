"""Parameter-efficient adapters for the Phase-1 linear layers.

LoRMA+ (Bihany, Patel, Modi 2025, arXiv 2506.07621, "additive inflation" variant, the best on average there):
    h = (alpha/r * B A + I) W0 x            B: (d_out, r), A: (r, d_out), W0 frozen
The low-rank matrix multiplies the frozen weight from the left (a transform of W0's output space) instead of being
added to it as in LoRA. Computed without forming any d x d matrix:  y = W0 x;  h = y + alpha/r * B (A y).
Init: B = 0, A random, so the model starts exactly at the pretrained weights.

LoRA (Hu et al. 2021) for comparison:  h = W0 x + alpha/r * B A x,  B: (d_out, r), A: (r, d_in).

rsLoRA (Kalajdzievski 2023, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"): LoRA with the scale
alpha / sqrt(r) instead of alpha / r, which keeps the update's magnitude stable as the rank grows.

BitFit (Ben Zaken et al. 2022): train only bias terms. Arch E's linear layers have no bias, so a zero-initialised bias
is added to every target linear (h = W0 x + b); the RMSNorm gains are trained too (the other non-matrix parameters).

IA3 (Liu et al. 2022, "Few-shot PEFT is better and cheaper than ICL"): learned vectors rescale the keys, the values
and the FFN intermediate activation:  k = l_k * W_k x,  v = l_v * W_v x,  down(l_ff * (silu(gate x) * up x)).
All l are initialised to 1, so the model starts exactly at the pretrained weights.
"""

import math

import torch.nn as nn


class LoRMAPlusLinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        d_out = base.out_features
        self.A = nn.Parameter(base.weight.new_empty(r, d_out))
        self.B = nn.Parameter(base.weight.new_zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * (self.drop(y) @ self.A.t() @ self.B.t())


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, rank_stabilized=False):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(base.weight.new_empty(r, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / math.sqrt(r) if rank_stabilized else alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.t() @ self.B.t())


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")


class BiasLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.bias = nn.Parameter(base.weight.new_zeros(base.out_features))

    def forward(self, x):
        y = self.base(x)
        return y + self.bias.to(y.dtype)          # keep autocast's bf16 (an fp32 bias upcasts everything after it)


class ScaleOut(nn.Module):
    """IA3 on a linear's output: l * W x."""
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(base.weight.new_ones(base.out_features))

    def forward(self, x):
        y = self.base(x)
        return y * self.scale.to(y.dtype)         # keep autocast's bf16 (fp32 k/v made attention run in fp32)


class ScaleIn(nn.Module):
    """IA3 on a linear's input: W (l * x) (for the FFN down projection: scales the intermediate activation)."""
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(base.weight.new_ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.scale.to(x.dtype))


def add_bitfit(backbone, targets=DEFAULT_TARGETS):
    """Freeze the backbone, add a trainable bias to every target linear and unfreeze the RMSNorm gains."""
    for p in backbone.parameters():
        p.requires_grad = False
    n = 0
    for block in backbone.blocks:
        for parent in (block.attn, block.ffn):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, BiasLinear(lin))
                    n += 1
    for m in backbone.modules():
        if isinstance(m, nn.RMSNorm):
            m.weight.requires_grad = True
    return n


def add_ia3(backbone):
    """Freeze the backbone, rescale keys, values (output of k_proj, v_proj) and the FFN intermediate (input of down)."""
    for p in backbone.parameters():
        p.requires_grad = False
    n = 0
    for block in backbone.blocks:
        block.attn.k_proj = ScaleOut(block.attn.k_proj)
        block.attn.v_proj = ScaleOut(block.attn.v_proj)
        block.ffn.down = ScaleIn(block.ffn.down)
        n += 3
    return n


class RSLoRALinear(LoRALinear):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__(base, r, alpha, dropout, rank_stabilized=True)


ADAPTERS = {"lorma": LoRMAPlusLinear, "lora": LoRALinear, "rslora": RSLoRALinear}


def add_adapters(backbone, kind, r=8, alpha=16, dropout=0.0, targets=DEFAULT_TARGETS):
    """Freeze every backbone weight, then wrap the target linear layers of every block with an adapter.
    Returns the number of wrapped layers. kind 'bitfit' / 'ia3' are delegated (r, alpha unused)."""
    if kind == "bitfit":
        return add_bitfit(backbone, targets)
    if kind == "ia3":
        return add_ia3(backbone)
    for p in backbone.parameters():
        p.requires_grad = False
    cls, n = ADAPTERS[kind], 0
    for block in backbone.blocks:
        for parent in (block.attn, block.ffn):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, cls(lin, r, alpha, dropout))
                    n += 1
    return n
