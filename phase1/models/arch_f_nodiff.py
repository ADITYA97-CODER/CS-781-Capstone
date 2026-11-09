"""Arch F: Arch E without differential attention (ablation of the differential-attention choice).

Everything else is identical to Arch E (arch_e_diff.py): hybrid local(512)/global layers in the
"alternate" pattern, grouped-query attention 12q / 4kv of head_dim 64, QK-norm, SwiGLU, tied
embeddings, doc_mask from scratch. The only change is that every layer uses standard softmax
attention (diff_layers is fixed to "none"), so there are no lambda vectors and no per-head norm.
Q/K/V/O projection shapes equal Arch E's, so the two models differ by only the 20 x (4 * 64 +
6 * 128) = 20,480 differential-attention parameters (Arch E 144.56M).
"""

from dataclasses import dataclass

from .arch_e_diff import ArchEConfig, ArchEDiff


@dataclass
class ArchFConfig(ArchEConfig):
    arch: str = "F"
    diff_layers: str = "none"

    def __post_init__(self):
        assert self.diff_layers == "none", "Arch F is Arch E without differential attention"
        super().__post_init__()


class ArchFNoDiff(ArchEDiff):
    def lambda_values(self):          # no differential layers: nothing to log
        return []
