from functools import partial
from importlib import import_module
from typing import Any, Mapping, Optional, Sequence

import torch

from ..core.categories import ModuleCategory
from ..core.registry import register
from ._base import _BaseLayer


@register("ArbitraryModule", inputs=["*"], outputs=["*"], category=ModuleCategory.ARBITRARY)
class ArbitraryModule(_BaseLayer, torch.nn.Module):
    """
    Wraps any importable callable so the rest of the pipeline can
    treat it like a normal Klay layer.
    """

    def __init__(
        self,
        target: str,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()

        args = args or []
        kwargs = kwargs or {}

        # run it and let it fail if module not found
        module, target = target.rsplit(".", 1)
        base_module = import_module(module)
        target_obj = getattr(base_module, target)
        try:
            is_nn_module = issubclass(target_obj, torch.nn.Module)
        except TypeError:
            print(f"Treating {target} as functional object")
            is_nn_module = False

        if isinstance(target_obj, torch.nn.Module):
            self.module = target_obj
        elif isinstance(target_obj, type) and issubclass(target_obj, torch.nn.Module):
            self.module = target_obj(*args, **kwargs)
        else:  # functional or other callable
            self.module = partial(target_obj, *args, **kwargs) if (args or kwargs) else target_obj

    def forward(self, *xs, **kw):
        return self.module(*xs, **kw)

    def __repr__(self):
        return f"ArbitraryModule({self.module!r})"

    @classmethod
    def from_config(cls, target: str, **cfg) -> "ArbitraryModule":
        return cls(target, **cfg)
