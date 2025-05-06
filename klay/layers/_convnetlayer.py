from typing import Any, Callable, Dict, List

import torch
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps

from ..core import ModuleCategory, register
from ..utils import irreps_blocks_to_string, tp_path_exists
from ._base import _BaseLayer
from ._interaction_block import InteractionBlock
from ._non_linear import ShiftedSoftPlus

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}


@register(
    "ConvNetLayer",
    inputs=["x", "h", "edge_length_embeddings", "edge_sh", "edge_index"],
    outputs=["h"],
    category=ModuleCategory.CONVOLUTION,
)
class ConvNetLayer(_BaseLayer, torch.nn.Module):
    """
    Args:

    """

    resnet: bool

    def __init__(
        self,
        irreps_in,  # its node embedding
        feature_irreps_hidden,
        node_attr_irreps,
        edge_attr_irreps,
        edge_embedding_irreps,
        convolution_kwargs: dict = {},
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "abs"},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = feature_irreps_hidden
        self.resnet = resnet

        self.irreps_in = irreps_in

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0 and tp_path_exists(irreps_in, edge_attr_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0 and tp_path_exists(irreps_in, edge_attr_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = "0e" if tp_path_exists(irreps_in, edge_attr_irreps, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        # TODO: partial resnet?
        if irreps_layer_out == irreps_in and resnet:
            # We are doing resnet updates and can for this layer
            self.resnet = True
        else:
            self.resnet = False

        # override defaults for irreps:
        self.conv = InteractionBlock(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            node_attr_irreps=node_attr_irreps,
            edge_attr_irreps=edge_attr_irreps,
            edge_embedding_irreps=edge_embedding_irreps,
            **convolution_kwargs,
        )
        # The output features are whatever we got in
        # updated with whatever the convolution outputs (which is a full graph module)
        self.irreps_out = self.equivariant_nonlin.irreps_out

    def forward(self, x, h, edge_length_embeddings, edge_sh, edge_index):
        # save old features for resnet
        old_h = h
        # run convolution
        h = self.conv(x, h, edge_length_embeddings, edge_sh, edge_index)
        # do nonlinearity
        h = self.equivariant_nonlin(h)

        # do resnet
        if self.resnet:
            h = old_h + h
        return h

    @classmethod
    def from_config(
        cls,
        hidden_irreps_lmax: int,
        edge_sh_lmax: int,
        conv_feature_size: int,
        input_block: List[dict[str, Any]],
        node_attr_block: List[dict[str, Any]],
        avg_neigh: int = 1,
        num_radial_basis: int = 8,
        convolution_kwargs: dict = {},
        resnet: bool = False,
        parity: bool = True,
        radial_network_hidden_dim=64,
        radial_network_layers=2,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "abs"},
    ):
        """Create a new instance from the config.

        Args:
            hidden_irreps_lmax (int): Maximum l value for hidden irreps.
            edge_sh_lmax (int): Maximum l value for edge spherical harmonics.
            conv_feature_size (int): Size of the convolution feature.
            input_block (List[dict[str, Any]]): Input block configuration.
            node_attr_block (List[dict[str, Any]]): Node attribute block configuration.
            avg_neigh (int, optional): Average number of neighbors. Defaults to 1.
            num_radial_basis (int, optional): Number of radial basis functions. Defaults to 8.
            convolution_kwargs (dict, optional): Additional convolution parameters. Defaults to {}.
            resnet (bool, optional): Whether to use residual connections. Defaults to False.
            parity (bool, optional): Whether to use parity. Defaults to True.
            radial_network_hidden_dim (int, optional): Hidden dimension for radial network. Defaults to 64.
            radial_network_layers (int, optional): Number of layers in the radial network. Defaults to 2.
            nonlinearity_type (str, optional): Type of nonlinearity. Defaults to "gate".
            nonlinearity_scalars (Dict[int, Callable], optional): Nonlinearity scalars. Defaults to {"e": "silu", "o": "tanh"}.
            nonlinearity_gates (Dict[int, Callable], optional): Nonlinearity gates. Defaults to {"e": "silu", "o": "abs"}.
        """
        node_embedding_irreps_in = irreps_blocks_to_string(input_block)
        conv_hidden_irrep = Irreps(
            [
                (conv_feature_size, (l, p))
                for p in ((1, -1) if parity else (1,))
                for l in range(hidden_irreps_lmax + 1)
            ]
        )
        edge_sh_irrep = Irreps.spherical_harmonics(edge_sh_lmax)

        conv_kw = {
            "invariant_layers": radial_network_layers,
            "invariant_neurons": radial_network_hidden_dim,
            "avg_num_neighbors": avg_neigh,
        }

        node_attr_irrep = irreps_blocks_to_string(node_attr_block)
        edge_embedding_irrep = Irreps(f"{num_radial_basis}x0e")

        return cls(
            node_embedding_irreps_in,
            conv_hidden_irrep,
            node_attr_irrep,
            edge_sh_irrep,
            edge_embedding_irrep,
            convolution_kwargs=conv_kw,
            resnet=resnet,
            nonlinearity_type=nonlinearity_type,
            nonlinearity_scalars=nonlinearity_scalars,
            nonlinearity_gates=nonlinearity_gates,
        )
