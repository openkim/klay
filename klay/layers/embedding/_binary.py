import torch
import torch.nn.functional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from ...registry import ModuleCategory, register
from .._base import _BaseLayer


@register(
    "BinaryAtomicNumberEncoding",
    inputs=["x"],
    outputs=["representation"],
    category=ModuleCategory.EMBEDDING,
)
class BinaryAtomicNumberEncoding(_BaseLayer, torch.nn.Module):
    """
    Compute a binary encoding of atoms' discrete atomic numbers.
    and returns a 1-0 encoding of atomic numbers of width 8.
    """

    def __init__(self):
        super().__init__()
        self.irreps_out = Irreps([(8, (0, 1))])

    def forward(self, x):
        representation = torch.zeros(x.size(0), 8, dtype=x.dtype, device=x.device)
        for i in range(7, -1, -1):
            representation[:, i] = x % 2
            x = torch.div(x, 2, rounding_mode="trunc")
        return representation

    @classmethod
    def from_config(cls):
        return cls()
