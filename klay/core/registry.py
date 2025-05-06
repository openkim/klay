from typing import Dict, List, Optional, Type

from torch import nn

from .categories import ModuleCategory
from .node import NodeMeta

_REGISTRY: Dict[str, NodeMeta] = {}


def register(
    name: str,
    *,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    category: Optional[ModuleCategory] = None,
):
    """
    Register a layer in the registry.
    :param name: layer name
    :param inputs: list of input port names (order matters)
    :param outputs: list of output port names (order matters)
    :param category: module category (e.g. "CONVOLUTION", "POOLING")
    :return: decorator
    """

    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise RuntimeError(f"Key '{name}' already registered")
        meta = NodeMeta(
            cls=cls,
            inputs=inputs or ["x"],  # sensible default
            outputs=outputs or ["x"],
            category=category,
        )
        _REGISTRY[name] = meta
        return cls

    return wrapper


def get(name: str) -> NodeMeta:
    """
    Get a layer by name from the registry.

    :param name:
    :return: Node metadata
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise RuntimeError(f"Unknown layer '{name}'. Available: {sorted(_REGISTRY)}") from exc


def names() -> List[str]:
    """
    Get all registered layer names.

    :return: sorted list of layer names
    """
    return sorted(_REGISTRY)


def build(name: str, **cfg):
    """Instantiate a registered layer via its from_config or __init__."""
    meta = get(name)  # NodeMeta
    cls = meta.cls
    if hasattr(cls, "from_config"):
        return cls.from_config(**cfg)
    return cls(**cfg)
