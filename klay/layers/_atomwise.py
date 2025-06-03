import logging
from typing import Any, List, Optional

import torch
import torch.nn.functional
from e3nn import nn
from e3nn.o3 import Irreps, Linear
from torch_runstats.scatter import scatter

from ..core import ModuleCategory, register
from ..utils import irreps_blocks_to_string
from ._base import _BaseLayer

# class AtomwiseOperation(torch.nn.Module):
#     def __init__(self, operation, field: str, irreps_in=None):
#         super().__init__()
#         self.operation = operation
#         # self.field = field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             my_irreps_in={field: operation.irreps_in},
#             irreps_out={field: operation.irreps_out},
#         )

#     def forward(self, data):
#         data = self.operation(data)
#         return data


@register("AtomwiseLinear", inputs=["h"], outputs=["h"], category=ModuleCategory.LINEAR)
class AtomwiseLinear(_BaseLayer, torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
    ):
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        super().__init__()
        self.linear = Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)

    def forward(self, h):
        print(h.dtype, self.linear)
        h = self.linear(h)
        return h

    @classmethod
    def from_config(
        cls, irreps_in_block: List[dict[str, Any]], irreps_out_block: List[dict[str, Any]]
    ):
        """Create a new instance from the config.

        Example usage:
            atomwise_linear = AtomwiseLinear.from_config(
                irreps_in_block=[{"l": 0, "p": odd, "mul": 10}],
                irreps_out_block=[{"l": 0, "p": odd, "mul": 1}],
            )
            irreps_in = "0x10e", irreps_out = "0x1e", or a 10x1 Linear layer

        Args:
            irreps_in_block: Irreps block for input
            irreps_out_block: Irreps block for output
        """
        irreps_in = irreps_blocks_to_string(irreps_in_block)
        irreps_out = irreps_blocks_to_string(irreps_out_block)
        return cls(irreps_in=irreps_in, irreps_out=irreps_out)


@register(
    "AtomwiseSumIndex", inputs=["src", "index"], outputs=["h"], category=ModuleCategory.LINEAR
)
class AtomwiseSumIndex(_BaseLayer, torch.nn.Module):
    """
    Scatter-sum along the first dimension.
      - model.train() -> full scatter-summed tensor
      - model.eval()  -> only the first row
    Fully TorchScript-compatible.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if index.dtype != torch.long:
            raise RuntimeError("index must have dtype torch.long")

        if index.dim() < src.dim():
            for _ in range(src.dim() - index.dim()):
                index = index.unsqueeze(-1)  # Script-compatible loop

        dim_size = int(index.max().item()) + 1

        out_shape = list(src.shape)
        out_shape[0] = dim_size
        reduced_out = src.new_zeros(out_shape)
        reduced_out.scatter_add_(0, index, src)

        return reduced_out

    @classmethod
    def from_config(cls) -> "AtomwiseSumIndex":
        return cls()


@register("KIMAPISumIndex", inputs=["src", "index"], outputs=["h"], category=ModuleCategory.LINEAR)
class KIMAPISumIndex(_BaseLayer, torch.nn.Module):
    """
    Wrapper around AtomwiseSumIndex.

    It is crafted specifically for KIM-APPI based applications, where at training time,
    it will sum like batch index, while at evaluation time, it will return the
    idx:0 contributing atoms energies
    """

    def __init__(self) -> None:
        super().__init__()
        self.scatter_sum = AtomwiseSumIndex()

    def forward(self, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:

        reduced_out = self.scatter_sum(src, index)

        if self.training:
            return reduced_out  # training -> full tensor
        else:
            return reduced_out[0]  # eval -> runtime in kim-api, contributing enegry

    @classmethod
    def from_config(cls) -> "KIMAPISumIndex":
        return cls()


@register("GatedAtomwiseLinear", inputs=["h"], outputs=["h"], category=ModuleCategory.LINEAR)
class GatedAtomwiseLinear(_BaseLayer, torch.nn.Module):
    """
    Linear layer with Gated Nonlinearity. It is reused a lot, hence adding it here
    for ease of use.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        act_scalars=torch.nn.functional.silu,
        act_gates=torch.nn.functional.silu,
    ):
        super().__init__()

        irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_in if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_in if ir.l > 0])

        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        irreps_hidden = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.lin1 = Linear(irreps_in, irreps_hidden)

        self.gate = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[act_scalars],
            irreps_gates=irreps_gates,
            act_gates=[act_gates],
            irreps_gated=irreps_gated,
        )
        self.lin2 = Linear(self.gate.irreps_out, irreps_out)
        self.irreps_out = irreps_out

    def forward(self, h):
        h = self.lin2(self.gate(self.lin1(h)))
        return h

    @classmethod
    def from_config(cls, irreps_in_block, irreps_out_block) -> "GatedAtomwiseLinear":
        irreps_in = Irreps(irreps_blocks_to_string(irreps_in_block))
        irreps_out = Irreps(irreps_blocks_to_string(irreps_out_block))
        return cls(irreps_in=irreps_in, irreps_out=irreps_out)


# class AtomwiseReduce(torch.nn.Module):
#     constant: float

#     def __init__(
#         self,
#         field: str,
#         out_field: Optional[str] = None,
#         reduce="sum",
#         avg_num_atoms=None,
#         irreps_in={},
#     ):
#         super().__init__()
#         assert reduce in ("sum", "mean", "normalized_sum")
#         self.constant = 1.0
#         if reduce == "normalized_sum":
#             assert avg_num_atoms is not None
#             self.constant = float(avg_num_atoms) ** -0.5
#             reduce = "sum"
#         self.reduce = reduce
#         self.field = field
#         self.out_field = f"{reduce}_{field}" if out_field is None else out_field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             irreps_out={self.out_field: irreps_in[self.field]}
#             if self.field in irreps_in
#             else {},
#         )

#     def forward(self, h, contrib):
#         E = scatter(h, contrib, dim=0, reduce=self.reduce)
#         if self.constant != 1.0:
#             E = E * self.constant
#         return E


# class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
#     """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

#     Args:
#         field: the per-atom field to scale/shift.
#         num_types: the number of types in the model.
#         shifts: the initial shifts to use, one per atom type.
#         scales: the initial scales to use, one per atom type.
#         arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
#             units (in which case they will be rescaled appropriately by any global rescaling later
#             applied to the model); if ``False``, the provided shifts/scales will be used without modification.

#             For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
#             But if scales/shifts computed from the training data are used, and are thus in dataset units,
#             this should be ``True``.
#         out_field: the output field; defaults to ``field``.
#     """

#     field: str
#     out_field: str
#     scales_trainble: bool
#     shifts_trainable: bool
#     has_scales: bool
#     has_shifts: bool

#     def __init__(
#         self,
#         field: str,
#         num_types: int,
#         type_names: List[str],
#         shifts: Optional[List[float]],
#         scales: Optional[List[float]],
#         arguments_in_dataset_units: bool,
#         out_field: Optional[str] = None,
#         scales_trainable: bool = False,
#         shifts_trainable: bool = False,
#         irreps_in={},
#     ):
#         super().__init__()
#         self.num_types = num_types
#         self.type_names = type_names
#         self.field = field
#         self.out_field = f"shifted_{field}" if out_field is None else out_field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
#             irreps_out={self.out_field: irreps_in[self.field]},
#         )

#         self.has_shifts = shifts is not None
#         if shifts is not None:
#             shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
#             if len(shifts.reshape([-1])) == 1:
#                 shifts = torch.ones(num_types) * shifts
#             assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
#             self.shifts_trainable = shifts_trainable
#             if shifts_trainable:
#                 self.shifts = torch.nn.Parameter(shifts)
#             else:
#                 self.register_buffer("shifts", shifts)

#         self.has_scales = scales is not None
#         if scales is not None:
#             scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
#             if len(scales.reshape([-1])) == 1:
#                 scales = torch.ones(num_types) * scales
#             assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
#             self.scales_trainable = scales_trainable
#             if scales_trainable:
#                 self.scales = torch.nn.Parameter(scales)
#             else:
#                 self.register_buffer("scales", scales)

#         self.arguments_in_dataset_units = arguments_in_dataset_units

#     def forward(self, x, h):

#         if not (self.has_scales or self.has_shifts):
#             return h

#         assert len(h) == len(x), "h doesnt seem to have correct per-atom shape"
#         if self.has_scales:
#             h = self.scales[x].view(-1, 1) * h
#         if self.has_shifts:
#             h = self.shifts[x].view(-1, 1) + h
#         return h

#     def update_for_rescale(self, rescale_module):
#         if hasattr(rescale_module, "related_scale_keys"):
#             if self.out_field not in rescale_module.related_scale_keys:
#                 return
#         if self.arguments_in_dataset_units and rescale_module.has_scale:
#             logging.debug(
#                 f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
#                 f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
#                 f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
#             )
#             with torch.no_grad():
#                 if self.has_scales:
#                     self.scales.div_(rescale_module.scale_by)
#                 if self.has_shifts:
#                     self.shifts.div_(rescale_module.scale_by)
#             logging.debug(
#                 f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
#                 f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
#             )
