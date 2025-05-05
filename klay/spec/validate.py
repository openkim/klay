# builder/validate.py
from pprint import pformat
from typing import Dict, List, Set


class ConfigValidationError(RuntimeError):
    pass


def validate_config(cfg: Dict):
    """Raise if the config is ill-formed or internally inconsistent."""
    required_top = {"model_inputs", "model_layers", "model_outputs"}
    missing = required_top - set(cfg)
    if missing:
        raise ConfigValidationError(f"Missing required sections: {missing}")

    inputs = set(cfg["model_inputs"])
    layers = cfg["model_layers"]
    outputs = cfg["model_outputs"]

    # 1. every layer declares a type
    for name, spec in layers.items():
        if "type" not in spec:
            raise ConfigValidationError(f"Layer '{name}' has no 'type' field.")

    # 2. gather all valid sources
    defined: Set[str] = set(inputs)
    for layer_name in layers:
        defined.add(layer_name)

    # 3. every edge source exists
    def _check_mapping(mapping: Dict, ctx: str, node: str):
        for src in mapping.values():
            root = src.split(".")[0]  # e.g. "initial_inputs.atomic_numbers"
            if root == "initial_inputs":
                root = src.split(".")[1]
            if root not in defined:
                raise ConfigValidationError(f"{ctx} '{node}' refers to unknown source '{root}'.")

    for lname, spec in layers.items():
        _check_mapping(spec.get("inputs", {}), "Layer", lname)

    for out in outputs:
        if out["source"] not in defined:
            raise ConfigValidationError(
                f"Model output '{out['name']}' refers to unknown source '{out['source']}'"
            )

    # 4. warn about dangling layers (present in graph, but not on any path to outputs)
    used: Set[str] = {out["source"] for out in outputs}

    # walk backwards
    queue: List[str] = list(used)
    while queue:
        current = queue.pop()
        if current in layers:
            for src in layers[current].get("inputs", {}).values():
                parent = src.split(".")[0] if "." not in src else src.split(".")[0]
                if parent == "initial_inputs":
                    parent = src.split(".")[1]
                if parent not in used:
                    used.add(parent)
                    queue.append(parent)

    dangling = set(layers) - used
    if dangling:
        print("[validate] Dangling layers that will never be reached:", pformat(dangling))
