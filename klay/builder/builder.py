from __future__ import annotations

from typing import Any, Dict

from ..core import build as _build_layer
from .fx_builder import build_fx_model


def _instantiate_layers(layer_cfg: Dict[str, Any]):
    return {
        n: _build_layer(spec["type"], **spec.get("config", {})) for n, spec in layer_cfg.items()
    }


def build_layers(cfg: Dict[str, Any]):
    """
    - If cfg defines inputs & outputs -> GraphModule
    - Else -> {name: nn.Module}
    """
    if "model_inputs" in cfg and "model_outputs" in cfg:
        return build_fx_model(cfg)
    return _instantiate_layers(cfg["model_layers"])
