import networkx as nx
import torch.fx as fx
from torch import nn

from klay.registry import NodeMeta
from klay.registry import get as get_layer


def build_dag(spec_dict):
    graph: nx.DiGraph = spec_dict.pop("_graph")  # produced by validator
    modules = {}  # name -> nn.Module instance
    for idx, node in enumerate(graph):  # topo order guaranteed
        layer_spec = spec_dict["layers"][idx]
        meta: NodeMeta = get_layer(layer_spec["type"])
        LayerCls = meta.cls
        modules[node] = LayerCls(**layer_spec.get("kwargs", {}))

    # 1) create a real nn.Module to host sub-modules
    class Model(nn.Module):
        pass

    model = Model()
    for name, mod in modules.items():
        model.add_module(name, mod)

    # 2) build fx.Graph that wires everything
    fxg = fx.Graph()
    node_map = {}
    placeholders = {}

    # helper to fetch or create placeholder for each source node
    def as_arg(src):
        if src not in node_map:
            ph = fxg.placeholder(src)
            placeholders[src] = ph
            node_map[src] = ph
        return node_map[src]

    for tgt in nx.topological_sort(graph):
        if tgt in placeholders:  # it was an input node
            continue
        layer_spec = next(
            l
            for l in spec_dict["layers"]
            if l.get("name", None) == tgt or f'{l["type"]}_{spec_dict["layers"].index(l)}' == tgt
        )

        inp_names = layer_spec.get("inputs") or list(graph.predecessors(tgt))
        args = tuple(as_arg(s) for s in inp_names)
        node_map[tgt] = fxg.call_module(tgt, args=args)

    fxg.output(node_map[tgt])  # last node is assumed output

    model.forward = fxg.compile_fn(model)  # dynamic forward

    return model
