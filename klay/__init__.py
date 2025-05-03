# klay/__init__.py
from importlib import import_module
from pathlib import Path
from typing import Union

_layers_root = Path(__file__).with_suffix("").parent / "layers"

for py in _layers_root.rglob("*.py"):  # <— recursive *.py search
    if py.name == "__init__.py":
        continue  # skip package markers
    rel_parts = py.relative_to(_layers_root).with_suffix("").parts
    module_name = "klay.layers." + ".".join(rel_parts)
    import_module(module_name)

from .builder.chain import build_chain
from .builder.dag import build_dag
from .registry import get as get_layer
from .registry import names as layer_names

# klay/__init__.py  (bottom)
from .spec import load_spec

# from .spec.yaml_loader import load_yaml
# from .builder.chain import build_chain
#
#
# def from_yaml(path: Union[str, Path]):
#     """Return nn.Sequential constructed from a simple list‐of‐layers YAML."""
#     spec = load_yaml(path)
#     return build_chain(spec)


def from_yaml(path):
    spec = load_spec(path)
    graph = spec.pop("_graph")  # already built by validator
    if graph.number_of_edges() == len(graph) - 1:
        return build_chain(spec)
    return build_dag(spec)


# def from_yaml(path):
#     spec = load_spec(path)
#     return build_chain(spec)
