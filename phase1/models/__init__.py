"""Phase-1 model registry: the config's "arch" field selects the architecture file.

  "E"  arch_e_diff.py      hybrid local/global decoder with differential attention, tied embeddings
  "F"  arch_f_nodiff.py    Arch E without differential attention (controlled ablation)

arch_d_sliding.py is kept because Arch E inherits its sliding-window machinery.
"""

from dataclasses import asdict

from .arch_e_diff import ArchEConfig, ArchEDiff
from .arch_f_nodiff import ArchFConfig, ArchFNoDiff

ARCHS = {
    "E": (ArchEConfig, ArchEDiff),
    "F": (ArchFConfig, ArchFNoDiff),
}


def build_model(cfg_dict):
    config_cls, model_cls = ARCHS[cfg_dict["arch"]]
    cfg = config_cls(**cfg_dict)
    return model_cls(cfg), cfg


def config_dict(cfg):
    return asdict(cfg)
