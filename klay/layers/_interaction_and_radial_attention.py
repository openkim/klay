"""Interaction Block"""

import math
from typing import Any, Callable, Dict, Optional

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct
from torch_runstats.scatter import scatter

from ..core import ModuleCategory, register
from ._base import _BaseLayer
from ._non_linear import ShiftedSoftPlus


@register(
    "AttentionInteractionBlock",
    inputs=["x", "h", "edge_length_embeddings", "edge_sh", "edge_index", "r_ijs"],
    outputs=["h"],
    category=ModuleCategory.ATTENTION,
)
class AttentionInteractionBlock(_BaseLayer, torch.nn.Module):
    avg_num_neighbors: Optional[float]
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        node_attr_irreps,
        edge_attr_irreps,
        edge_embedding_irreps,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:
        """
        InteractionBlock.

        :param irreps_node_attr: Nodes attribute irreps
        :param irreps_edge_attr: Edge attribute irreps
        :param irreps_out: Output irreps, in our case typically a single scalar
        :param radial_layers: Number of radial layers, default = 1
        :param radial_neurons: Number of hidden neurons in radial function, default = 8
        :param avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        :param number_of_basis: Number or Basis function, default = 8
        :param irreps_in: Input Features, default = None
        :param use_sc: bool, use self-connection or not
        """
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        feature_irreps_in = irreps_in
        feature_irreps_out = irreps_out
        irreps_edge_attr = edge_attr_irreps

        # - Build modules -
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [edge_embedding_irreps.num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.tp = tp

        self.linear_2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                node_attr_irreps,
                feature_irreps_out,
            )
        self.irreps_out = feature_irreps_out

        self.radial_attention = RadialAttention(n_input=8, n_hidden_layers=2, hidden_layer_width=8)

    def forward(self, x, h, edge_length_embeddings, edge_sh, edge_index, r_ijs):
        weight = self.fc(edge_length_embeddings)

        # x = h
        edge_src = edge_index[1]
        edge_dst = edge_index[0]

        if self.sc is not None:
            sc = self.sc(h, x)

        h = self.linear_1(h)
        edge_features = self.tp(h[edge_src], edge_sh, weight)
        attention = self.radial_attention(r_ijs)
        h = scatter(edge_features * attention, edge_dst, dim=0, dim_size=len(h))

        h = self.linear_2(h)

        if self.sc is not None:
            h = h + sc
        return h

    @classmethod
    def from_config(
        cls,
        irreps_in,
        irreps_out,
        node_attr_irreps,
        edge_attr_irreps,
        edge_embedding_irreps,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> "AttentionInteractionBlock":
        """
        Create an AttentionInteractionBlock from configuration.
        """
        cls(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            node_attr_irreps=node_attr_irreps,
            edge_attr_irreps=edge_attr_irreps,
            edge_embedding_irreps=edge_embedding_irreps,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            avg_num_neighbors=avg_num_neighbors,
            use_sc=use_sc,
            nonlinearity_scalars=nonlinearity_scalars,
        )


@register(
    "RadialAttention", inputs=["r_ijs"], outputs=["attention"], category=ModuleCategory.ATTENTION
)
class RadialAttention(_BaseLayer, torch.nn.Module):
    """
    Scalar radial attention mechanism.
    TODO: Equivariant attention?
    """

    def __init__(
        self,
        n_input: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        nonlinearity: str = "silu",
    ):
        super().__init__()
        self.n_input = n_input
        self.register_buffer("n", torch.arange(1, n_input + 1, dtype=torch.float32) * math.pi)
        nn_layers = []
        nn_layers.append(torch.nn.Linear(n_input, hidden_layer_width))
        nn_layers.append(
            {
                "silu": torch.nn.functional.silu,
                "ssp": ShiftedSoftPlus,
            }[nonlinearity]()
        )
        for i in range(n_hidden_layers):
            nn_layers.append(torch.nn.Linear(hidden_layer_width, hidden_layer_width))
            nn_layers.append(
                {
                    "silu": torch.nn.functional.silu,
                    "ssp": ShiftedSoftPlus,
                }[nonlinearity]()
            )
        nn_layers.append(torch.nn.Linear(hidden_layer_width, 1))
        self.nn = torch.nn.Sequential(*nn_layers)

    def forward(self, r_ijs):
        inputs = torch.sin(self.n * r_ijs.unsqueeze(-1)) / r_ijs.unsqueeze(-1)
        attention = self.nn(inputs)
        return attention

    @classmethod
    def from_config(
        cls, n_input: int, n_hidden_layers: int, hidden_layer_width: int
    ) -> "RadialAttention":
        """Create a new instance from the config.

        Args:
            n_input (int): Number of input features.
            n_hidden_layers (int): Number of hidden layers.
            hidden_layer_width (int): Width of hidden layers.
        """
        return cls(
            n_input=n_input, n_hidden_layers=n_hidden_layers, hidden_layer_width=hidden_layer_width
        )
