# builder/dag.py
from typing import Dict

import networkx as nx


class ModelDAG:
    """Directed acyclic graph of data dependencies."""

    def __init__(self, cfg: Dict):
        self.graph = nx.DiGraph()
        self.inputs = cfg["model_inputs"]
        self.layers = cfg["model_layers"]
        self.outputs = cfg["model_outputs"]
        self._build(cfg)

    def _add_edge(self, src: str, dst: str, src_port: str):
        """Edge carries tensor named `src_port` from src ➜ dst."""
        self.graph.add_edge(src, dst, tensor=src_port)

    def _build(self, cfg: Dict):
        # Add input nodes
        for inp in self.inputs:
            self.graph.add_node(inp, kind="input")

        # Add layer nodes
        for lname, spec in self.layers.items():
            self.graph.add_node(lname, kind="layer", layer_spec=spec)

        # Wire them
        for lname, spec in self.layers.items():
            for port, src in (spec.get("inputs") or {}).items():
                root, *rest = src.split(".")
                if root == "initial_inputs":
                    root = rest[0]  # true input key
                self._add_edge(root, lname, port)

        # Add outputs as dummy sink nodes
        for out in self.outputs:
            name = f"__out_{out['name']}"
            self.graph.add_node(name, kind="output", output_name=out["name"])
            self._add_edge(out["source"], name, out["name"])

        # Check acyclicity
        if not nx.is_directed_acyclic_graph(self.graph):
            raise RuntimeError("Config yields cycles – cannot build a model.")


def build_dag(cfg: Dict) -> ModelDAG:
    """Build a DAG from the config."""
    dag = ModelDAG(cfg)
    return dag
