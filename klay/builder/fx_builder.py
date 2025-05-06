"""
Turn a NetworkX DAG into torch.fx.GraphModule.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, Optional

import networkx as nx
import torch.fx as fx
import torch.nn as nn

from ..core import build as build_layer
from .dag import build_dag, canonical_source


def build_fx_model(cfg: Dict[str, Any], dag: Optional[nx.DiGraph] = None) -> nn.Module:
    dag = dag or build_dag(cfg)
    layers_cfg = cfg["model_layers"]
    modules: Dict[str, nn.Module] = {
        lname: build_layer(spec["type"], **spec.get("config", {}))
        for lname, spec in layers_cfg.items()
    }

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
        kwargs = {}

        for edge_src, _, edge_data in dag.in_edges(n, data=True):
            port = edge_data["port"]  # layer arg name
            kwargs[port] = alias2node[edge_src]

        fx_node = g_fx.call_module(n, kwargs=kwargs)
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
