from pathlib import Path

import yaml

from .interpolate import interpolate_globals
from .validate import validate_graph, validate_spec
from .yaml_loader import load_yaml  # Phase 2 helper


def load_spec(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    raw = validate_spec(raw)
    raw = interpolate_globals(raw)
    graph = validate_graph(raw)
    raw["_graph"] = graph
    return raw
