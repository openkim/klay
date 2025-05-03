from dataclasses import dataclass
from typing import List, Optional, Type

from torch import nn

from .categories import ModuleCategory


# TODO: slots, with python > 3.9
@dataclass(frozen=True)
class NodeMeta:
    """
    Metadata for a layer class. This is used to register a layer in the registry.
    It contains the following fields:
    """

    cls: Type[nn.Module]  # the actual layer class
    inputs: List[str]  # canonical port names (order matters)
    outputs: List[str]  # canonical port names
    category: Optional[ModuleCategory] = None  # e.g. "CONVOLUTION", "POOLING"
