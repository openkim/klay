from typing import Optional, Union

import torch

from ..core import ModuleCategory, register
from ._base import _BaseLayer


@register(
    "AutogradForces", inputs=["energy", "pos"], outputs=["forces"], category=ModuleCategory.AUTOGRAD
)
class AutogradForces(_BaseLayer, torch.nn.Module):
    """Autograd forces layer.
    Computes the forces using autograd.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, energy: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad([energy.sum()], [pos], create_graph=True, retain_graph=True)[
            0
        ]
        if gradient is None:
            raise RuntimeError("Failed to compute forces.")
        else:
            forces = -gradient
        return forces

    @classmethod
    def from_config(cls) -> "AutogradForces":
        """Create a new instance from the config."""
        return cls()
