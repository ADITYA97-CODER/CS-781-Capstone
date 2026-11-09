"""Phase-1 model registry: the config's "arch" field selects the architecture file.

  "A"  arch_a_decoder.py   Llama-style causal decoder
  "C"  arch_c_hybrid.py    GPT-BERT hybrid (causal + bidirectional masked batches)
  "D"  arch_d_sliding.py   sliding-window + grouped-query efficient decoder
  "E"  arch_e_diff.py      hybrid local/global decoder with differential attention, tied embeddings
  "F"  arch_f_nodiff.py    Arch E without differential attention (ablation)
"""

from dataclasses import asdict

from .arch_a_decoder import ArchAConfig, ArchADecoder
from .arch_c_hybrid import ArchCConfig, ArchCHybrid
from .arch_d_sliding import ArchDConfig, ArchDSliding
from .arch_e_diff import ArchEConfig, ArchEDiff
from .arch_f_nodiff import ArchFConfig, ArchFNoDiff

ARCHS = {
    "A": (ArchAConfig, ArchADecoder),
    "C": (ArchCConfig, ArchCHybrid),
    "D": (ArchDConfig, ArchDSliding),
    "E": (ArchEConfig, ArchEDiff),
    "F": (ArchFConfig, ArchFNoDiff),
}


def build_model(cfg_dict):
    config_cls, model_cls = ARCHS[cfg_dict["arch"]]
    cfg = config_cls(**cfg_dict)
    return model_cls(cfg), cfg


def config_dict(cfg):
    return asdict(cfg)
