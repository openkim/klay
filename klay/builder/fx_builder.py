"""
Turn a NetworkX DAG into torch.fx.GraphModule.
"""

import operator
from typing import Any, Dict, Optional

import networkx as nx
import torch.fx as fx
import torch.nn as nn

from ..core import build as build_layer
from .dag import build_dag, canonical_source


def build_fx_model(cfg: Dict[str, Any], dag: Optional[nx.MultiDiGraph] = None) -> nn.Module:
    dag = dag or build_dag(cfg)
    layers_cfg = cfg["model_layers"]

    modules: Dict[str, nn.Module] = {}
    for lname, spec in layers_cfg.items():
        if "alias" not in spec:
            modules[lname] = build_layer(spec["type"], **spec.get("config", {}))

    # aliases
    for lname, spec in layers_cfg.items():
        if "alias" in spec:
            target = spec["alias"]
            if target not in modules:
                raise ValueError(f"Alias {lname} points to unknown layer {target}")
            modules[lname] = modules[target]

    g_fx = fx.Graph()
    alias2node = {}

    # inputs
    for n, data in dag.nodes(data=True):
        if data.get("kind") == "input":
            alias2node[n] = g_fx.placeholder(n)

    # layers (topo order)
    for n in nx.topological_sort(dag):
        kind = dag.nodes[n].get("kind")
        if kind not in ("layer", "alias"):
            continue

        spec = layers_cfg[n]
        pos_args: Dict[int, Any] = {}
        kw_args: Dict[str, Any] = {}

        for edge_src, _, edge_data in dag.in_edges(n, data=True):
            port = edge_data["port"]  # layer arg name

            if isinstance(port, int) or (isinstance(port, str) and port.isdigit()):
                pos_args[int(port)] = alias2node[edge_src]
            else:
                kw_args[str(port)] = alias2node[edge_src]

        ordered_args = tuple(v for k, v in sorted(pos_args.items()))

        fx_node = g_fx.call_module(n, args=ordered_args, kwargs=kw_args)
        alias2node[n] = fx_node

        # user-defined output aliases
        for real_out, alias in (spec.get("output") or {}).items():
            key = (
                int(real_out) if isinstance(real_out, int) or str(real_out).isdigit() else real_out
            )
            get = g_fx.call_function(operator.getitem, args=(fx_node, key))

            alias2node[alias] = get

    outs_cfg = cfg["model_outputs"]  # now a dict
    out_nodes = {}

    for pub_name, src in outs_cfg.items():
        layer_name, *maybe_port = src.split(".", 1)
        node = alias2node[layer_name]

        if maybe_port:  # need node[port]
            port = maybe_port[0]
            node = g_fx.call_function(operator.getitem, args=(node, port))

        out_nodes[pub_name] = node

    # single vs multiple outputs
    if len(out_nodes) == 1:
        g_fx.output(next(iter(out_nodes.values())))
    else:
        # preserve order given in YAML
        g_fx.output(tuple(out_nodes[k] for k in outs_cfg.keys()))
    gm = fx.GraphModule(modules, g_fx)
    g_fx.lint()
    gm.recompile()
    return gm
