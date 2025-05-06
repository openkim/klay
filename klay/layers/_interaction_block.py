"""Interaction Block"""

from typing import Callable, Dict, Optional

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct
from torch_runstats.scatter import scatter

from ..core import ModuleCategory, register
from ._base import _BaseLayer
from ._non_linear import ShiftedSoftPlus

# @torch.jit.script


@register(
    "InteractionBlock",
    inputs=["x", "h", "edge_length_embeddings", "edge_sh", "edge_index"],
    outputs=["h"],
    category=ModuleCategory.CONVOLUTION,
)
class InteractionBlock(_BaseLayer, torch.nn.Module):
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
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu"},
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

    def forward(self, x, h, edge_length_embeddings, edge_sh, edge_index):
        weight = self.fc(edge_length_embeddings)

        # x = h
        edge_src = edge_index[1]
        edge_dst = edge_index[0]

        if self.sc is not None:
            sc = self.sc(h, x)

        h = self.linear_1(h)
        edge_features = self.tp(h[edge_src], edge_sh, weight)
        h = scatter(edge_features, edge_dst, dim=0, dim_size=len(h))

        # Necessary to get TorchScript to be able to type infer when its not None
        avg_num_neigh: Optional[float] = self.avg_num_neighbors
        if avg_num_neigh is not None:
            h = h.div(avg_num_neigh**0.5)

        h = self.linear_2(h)

        if self.sc is not None:
            h = h + sc
        return h

    @classmethod
    def from_config(
        cls, irreps_in, irreps_out, node_attr_irreps, edge_attr_irreps, edge_embedding_irreps
    ):
        """
        Create an InteractionBlock from irreps.

        Note: Not putting much effort into this, as this will not be used normally.

        :param irreps_in: Input irreps
        :param irreps_out: Output irreps
        :param node_attr_irreps: Node attribute irreps
        :param edge_attr_irreps: Edge attribute irreps
        :param edge_embedding_irreps: Edge embedding irreps
        :return: InteractionBlock instance
        """
        return cls(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            node_attr_irreps=node_attr_irreps,
            edge_attr_irreps=edge_attr_irreps,
            edge_embedding_irreps=edge_embedding_irreps,
        )
