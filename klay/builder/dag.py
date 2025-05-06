"""
NetworkX-based DAG builder for MLIP configs.
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx


def build_dag(cfg: Dict[str, Any]) -> nx.DiGraph:
    """
    Build a directed acyclic graph (DAG) from the model configuration.
    It is used for sanity analysis mostly, as fx.graph can directly
    build a DAG from the model.
    """
    g = nx.DiGraph()
    inputs_cfg = cfg.get("model_inputs", {})
    layers_cfg = cfg["model_layers"]
    outputs_cfg = cfg.get("model_outputs", {})

    # 1. inputs
    for name in inputs_cfg:
        g.add_node(name, kind="input")

    # 2. layers & aliases
    declared: set[str] = set()  # owns weights
    for lname, spec in layers_cfg.items():
        if "alias" in spec:
            target = spec["alias"]
            g.add_node(lname, kind="alias", target=target, meta=spec)
        else:  # real module
            declared.add(lname)
            g.add_node(lname, kind="layer", meta=spec)

        # - edges from declared inputs
        for port, src in (spec.get("inputs") or {}).items():
            parent = canonical_source(src)
            g.add_edge(parent, lname, port=port)

        for real_key, alias in (spec.get("output") or {}).items():
            g.add_node(alias, kind="alias_out", parent=lname, key=real_key)  # meta for info
            g.add_edge(lname, alias, port=real_key)

            # node for layer.key (dot-index form)
            dot_name = f"{lname}.{real_key}"
            g.add_node(dot_name, kind="alias_out", parent=lname, key=real_key)
            g.add_edge(lname, dot_name, port=real_key)

    # 3. outputs
    for pub_name, src in outputs_cfg.items():
        sink = f"OUT::{pub_name}"
        layer_name = src.split(".", 1)[0]
        port = src.split(".", 1)[1] if "." in src else None
        g.add_node(sink, kind="output", expose=pub_name, port=port)
        g.add_edge(layer_name, sink)

    # 4. sanity checks
    _validate_graph(g, declared)
    return g


def canonical_source(ref: str) -> str:
    """'model_inputs.x' -> 'x',  'layer.port' -> 'layer',  'layer' -> 'layer'"""
    if ref.startswith("model_inputs."):
        return ref.split(".", 1)[1]
    return ref.split(".", 1)[0]


def _validate_graph(g: nx.DiGraph, declared_layers: set[str]):
    """
    Validate the dependency graph.
    """
    ghost = [n for n, d in g.nodes(data=True) if "kind" not in d]
    if ghost:
        raise ValueError(f"Unknown inputs / layers referenced: {sorted(ghost)}")

    bad_alias = [
        n
        for n, d in g.nodes(data=True)
        if d.get("kind") == "alias" and d["target"] not in declared_layers
    ]
    if bad_alias:
        raise ValueError(f"Aliases pointing to undeclared layers: {sorted(bad_alias)}")

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Dependency graph has cycles")
