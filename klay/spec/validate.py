from typing import Any, Dict, List, Set

import networkx as nx

from klay.registry import get as get_layer
from klay.registry import names as layer_names


def validate_graph(spec):
    names = []
    g = nx.DiGraph()

    for idx, l in enumerate(spec["layers"]):
        lname = l.get("name", f'{l["type"]}_{idx}')
        if lname in names:
            raise ValueError(f"duplicate layer name '{lname}'")
        names.append(lname)
        g.add_node(lname)

        # inputs default to previous layer
        inputs = l.get("inputs") or ([names[-2]] if idx else [])
        for src in inputs:
            if src not in names[:-1]:
                raise ValueError(f"layers[{idx}] refers to unknown input '{src}'")
            g.add_edge(src, lname)

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("cycle detected in layer graph")

    return g  # return the DiGraph for builder use


# klay/spec/validate.py
from typing import Any, Dict, List, Set

import networkx as nx

from klay.registry import names as layer_names


def validate_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    # ── 0. top-level sanity ───────────────────────────────────────────
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping")

    layers: List[dict] = raw.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("'layers' must be a non-empty list")

    # optional “global” block
    if "global" in raw and not isinstance(raw["global"], dict):
        raise ValueError("'global' must be a mapping")

    # ── 1. per-layer checks + build dependency list ──────────────────
    seen_names: Set[str] = set()
    graph_edges = []  # (src, dst) tuples

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{idx}] must be a mapping")

        # 1a. type exists in registry
        ltype = layer.get("type")
        if ltype not in layer_names():
            raise ValueError(
                f"layers[{idx}].type='{ltype}' is unknown; "
                f"available: {', '.join(layer_names())}"
            )

        # 1b. kwargs mapping if present
        if "kwargs" in layer and not isinstance(layer["kwargs"], dict):
            raise ValueError(f"layers[{idx}].kwargs must be a mapping")

        # 1c. determine this layer’s unique name
        lname = layer.get("name", f"{ltype}_{idx}")
        if lname in seen_names:
            raise ValueError(f"duplicate layer name '{lname}' (idx {idx})")
        seen_names.add(lname)
        layer["_auto_name"] = lname  # stash for builder

        # 1d. inputs list (if present) must reference *earlier* layers
        inputs = layer.get("inputs")
        if inputs is not None:
            if not (isinstance(inputs, list) and all(isinstance(s, str) for s in inputs)):
                raise ValueError(f"layers[{idx}].inputs must be a list of strings")
            for src in inputs:
                if src not in seen_names:
                    raise ValueError(f"layers[{idx}] refers to unknown or future layer '{src}'")
                graph_edges.append((src, lname))
        elif idx:  # implicit chain
            prev_name = layers[idx - 1].get("_auto_name")
            graph_edges.append((prev_name, lname))

    # ── 2. cycle detection ───────────────────────────────────────────
    g = nx.DiGraph()
    g.add_nodes_from(seen_names)
    g.add_edges_from(graph_edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("layers graph contains a cycle")

    raw["_graph"] = g  # for the DAG builder
    return raw
