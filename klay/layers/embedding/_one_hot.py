from typing import Any

import torch
import torch.nn.functional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from ...core import ModuleCategory, register
from ...utils.misc import get_torch_dtype
from .._base import _BaseLayer


@register(
    "OneHotAtomEncoding",
    inputs=["x"],
    outputs=["one_hot"],
    category=ModuleCategory.EMBEDDING,
)
@compile_mode("script")
class OneHotAtomEncoding(_BaseLayer, torch.nn.Module):
    """Compute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    def __init__(
        self,
        num_elems: int,
        input_is_atomic_number: bool = False,
    ):
        super().__init__()
        self.num_elems = num_elems
        # if input_is_atomic_number is True, the input is assumed to be atomic numbers
        # otherwise, it is assumed to be a 0-based index of the elements. For
        # the atomic numbers, we need to shift the one-hot encoding by 1 to account for the fact that
        # the first class in one hot is supposed to be 0, while the atomic numbers start from 1.

        z_to_idx_shift = torch.tensor(1) if input_is_atomic_number else torch.tensor(0)
        self.register_buffer("z_to_idx_shift", z_to_idx_shift)
        # Output irreps are num_elems even (invariant) scalars
        self.irreps_out = Irreps([(self.num_elems, (0, 1))])

    def forward(self, x):  # TODO input data type
        one_hot = torch.nn.functional.one_hot(x - self.z_to_idx_shift, num_classes=self.num_elems)
        # Convert to float
        one_hot = one_hot.to(torch.get_default_dtype())
        return one_hot

    @classmethod
    def from_config(
        cls, num_elems: int, input_is_atomic_number: bool = False
    ) -> "OneHotAtomEncoding":
        """Create a new instance from the config.

        Args:
            num_elems: Number of elements to encode.
            input_is_atomic_number: If ``True``, the input is assumed to be atomic numbers.
                Otherwise, it is assumed to be a 0-based index of the elements.
        """
        return cls(num_elems=num_elems, input_is_atomic_number=input_is_atomic_number)
