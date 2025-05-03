from pathlib import Path

import yaml

from .validate import validate_spec


def load_spec(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    raw = validate_spec(raw)
    return raw
