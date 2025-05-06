from typing import Any

import torch
from e3nn import nn, o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from torch_scatter import scatter

from ..core import ModuleCategory, register
from ..utils import irreps_blocks_to_string
from ._base import _BaseLayer


@register(
    "E3Attention",
    inputs=["f", "edge_index", "edge_length", "edge_sh", "edge_length_embedded"],
    outputs=["f_out"],
    category=ModuleCategory.ATTENTION,
)
class E3Attention(_BaseLayer, torch.nn.Module):
    """
    An SE(3)-equivariant attention module.
        https://docs.e3nn.org/en/latest/guide/transformer.html
    Implements formula (1) from "SE(3)-Transformers: 3D Roto-Translation
    Equivariant Attention Networks" (https://arxiv.org/abs/2006.10503).
    """

    def __init__(
        self,
        irreps_input: o3.Irreps,
        irreps_query: o3.Irreps,
        irreps_key: o3.Irreps,
        irreps_edge_sh: o3.Irreps,
        irreps_value: o3.Irreps,
        number_of_basis: int = 10,
        max_radius: float = 1.3,
        radial_neurons: int = 16,
    ):
        """
        Args:
            irreps_input (o3.Irreps): Irreps of the input features.
            irreps_query (o3.Irreps): Irreps of the query embeddings.
            irreps_key (o3.Irreps): Irreps of the key embeddings.
            irreps_value (o3.Irreps): Irreps of the value (also output) embeddings.
            number_of_basis (int): Number of radial basis functions.
            max_radius (float): Radius cutoff for the neighbor graph.
            radial_neurons (int): Hidden size for the radial MLP.
        """
        super().__init__()
        self.irreps_input = irreps_input
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_value = irreps_value

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.radial_neurons = radial_neurons
        self.irreps_sh = irreps_edge_sh

        # Q: linear from input -> query irreps
        self.h_q = o3.Linear(self.irreps_input, self.irreps_query)

        # K: (f x SH) -> K
        self.tp_k = o3.FullyConnectedTensorProduct(
            self.irreps_input, self.irreps_sh, self.irreps_key, shared_weights=False
        )
        self.fc_k = nn.FullyConnectedNet(
            [self.number_of_basis, radial_neurons, self.tp_k.weight_numel],
            act=torch.nn.functional.silu,
        )

        # V: (f x SH) -> V
        self.tp_v = o3.FullyConnectedTensorProduct(
            self.irreps_input, self.irreps_sh, self.irreps_value, shared_weights=False
        )
        self.fc_v = nn.FullyConnectedNet(
            [self.number_of_basis, radial_neurons, self.tp_v.weight_numel],
            act=torch.nn.functional.silu,
        )

        # Dot product (Q x K) -> scalar (0e)
        self.dot = o3.FullyConnectedTensorProduct(self.irreps_query, self.irreps_key, "0e")

        self.irreps_out = irreps_input

    def forward(
        self,
        f: torch.Tensor,
        edge_index: torch.tensor,
        edge_length: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length_embedded: torch.Tensor,
    ):

        # Q (per-node)
        q = self.h_q(f)

        edge_src, edge_dst = edge_index[0], edge_index[1]  # 0 -> 1
        # K, V (per-edge)
        k = self.tp_k(f[edge_dst], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        # Compute unnormalized attention scores
        # get cutoffs from the radial basis?
        edge_weight_cutoff = soft_unit_step(10.0 * (1.0 - edge_length / self.max_radius))
        exp_ijk = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()

        # Sum for normalization per destination node
        z = scatter(exp_ijk, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1.0  # avoid division by zero

        alpha = exp_ijk / z[edge_dst]

        # Example uses alpha.relu().sqrt() for numerical stability, but why?
        # Also sqrt(alpha) instead of alpha for normalization?
        f_out = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
        return f_out

    @classmethod
    def from_config(
        cls,
        irreps_input_block,
        irreps_query_block,
        irreps_key_block,
        edge_sh_lmax,
        irreps_value_block,
        number_of_basis: int = 10,
        max_radius: float = 1.3,
        radial_neurons: int = 16,
    ) -> Any:
        """Create a new instance from the config.

        Parameters:
            irreps_input_block: Irreps of the input features.
            irreps_query_block: Irreps of the query embeddings.
            irreps_key_block: Irreps of the key embeddings.
            edge_sh_lmax: l_max for spherical harmonics
            irreps_value_block: Irreps of the value (also output) embeddings.
            number_of_basis (int): Number of radial basis functions.
            max_radius (float): Radius cutoff for the neighbor graph.
            radial_neurons (int): Hidden size for the radial MLP.
        """
        irreps_input = o3.Irreps(irreps_blocks_to_string(irreps_input_block))
        irreps_query = o3.Irreps(irreps_blocks_to_string(irreps_query_block))
        irreps_key = o3.Irreps(irreps_blocks_to_string(irreps_key_block))
        irreps_value = o3.Irreps(irreps_blocks_to_string(irreps_value_block))

        irreps_edge_sh = o3.Irreps.spherical_harmonics(edge_sh_lmax)
        return cls(
            irreps_input=irreps_input,
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            irreps_edge_sh=irreps_edge_sh,
            irreps_value=irreps_value,
            number_of_basis=number_of_basis,
            max_radius=max_radius,
            radial_neurons=radial_neurons,
        )
