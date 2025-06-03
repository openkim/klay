from importlib import import_module, metadata
from pathlib import Path
from typing import Union

__version__ = metadata.version("klay")

_layers_root = Path(__file__).with_suffix("").parent / "layers"

for py in _layers_root.rglob("*.py"):  # <- recursive *.py search
    if py.name == "__init__.py":
        continue  # skip package markers
    rel_parts = py.relative_to(_layers_root).with_suffix("").parts
    module_name = "klay.layers." + ".".join(rel_parts)
    import_module(module_name)
