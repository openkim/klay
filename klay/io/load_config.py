from pathlib import Path
from typing import Union

from omegaconf import OmegaConf


def load_config(path: Union[str, Path]) -> dict:
    """OmegaConf -> plain Python dict, fully resolved (interpolations expanded)."""
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg
