# klay/builder/named_dag.py
import networkx as nx
import torch.fx as fx
from torch import nn

from ..registry import get as get_layer


def build_named_dag(spec: dict) -> nn.Module:
    layers = spec["model_layers"]
    G = nx.DiGraph()

    # 0. add initial_inputs node
    G.add_node("initial_inputs")

    # 1. build dependency graph
    for lname, ldata in layers.items():
        G.add_node(lname)
        for _, (src_layer, _) in ldata.get("inputs", {}).items():
            G.add_edge(src_layer, lname)

    # 2. topological order check
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("cycle in model_layers")

    # 3. create nn.Module shell + sub-modules
    class Model(nn.Module):
        pass

    model = Model()

    for lname in layers:
        meta = get_layer(layers[lname]["type"])
        params = layers[lname].get("params", {})
        mod = meta.cls(**params)
        model.add_module(lname, mod)

    # 4. build torch.fx graph
    fxg = fx.Graph()
    node_t = {}  # layer_name -> fx.Node

    # placeholders for each entry in initial_inputs
    ph_map = {}
    for port in spec["initial_inputs"]:
        ph_map[port] = fxg.placeholder(port)

    node_t["initial_inputs"] = ph_map  # treat as dict of ports

    # iterate topologically
    for lname in nx.topological_sort(G):
        if lname == "initial_inputs":
            continue

        meta = get_layer(layers[lname]["type"])
        port_map = layers[lname]["inputs"]
        # assemble args in canonical order
        args = []
        for p in meta.inputs:
            src_layer, src_port = port_map[p]
            if src_layer == "initial_inputs":
                arg = node_t[src_layer][src_port]
            else:
                arg = node_t[src_layer]  # output handle
                if src_port is not None:
                    arg = fxg.call_function(
                        getattr, args=(arg, src_port)
                    )  # optional field extraction
            args.append(arg)

        node_t[lname] = fxg.call_module(lname, tuple(args))

    # assume single scalar output for brevity
    fxg.output(node_t[spec["model_outputs"][0]["source"]])

    model.forward = fxg.compile_fn(model)
    return model
