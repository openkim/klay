from collections import OrderedDict

from torch import nn

from klay.registry import get as get_layer


def build_chain(spec_dict: dict) -> nn.Sequential:
    ordered = OrderedDict()
    for idx, layer_cfg in enumerate(spec_dict["layers"]):
        meta = get_layer(layer_cfg["type"])  # NodeMeta
        LayerCls = meta.cls
        kwargs = layer_cfg.get("kwargs", {})
        mod_name = layer_cfg.get("name", f"{layer_cfg['type']}_{idx}")
        ordered[mod_name] = LayerCls(**kwargs)
    return nn.Sequential(ordered)
