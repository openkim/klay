import re
from typing import List

import torch
from e3nn.o3 import Irreps

NEQUIP_BLOCK = """
class NequipConvBlock(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        conv_layers: List["ConvNetLayer"],
    ):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.irreps_out = self.conv_layers[-1].irreps_out

    def forward(self,x, h,
        <<edge_length_embeddings>>
        <<edge_sh>>
        <<edge_index>>
        ):
<<CONV>>
        return h
"""


def get_nequip_conv(
    parity: bool,
    lmax: int,
    conv_feature_size: int,
    node_embedding_irrep_in,
    node_attr_irrep,
    edge_attr_irrep,
    edge_embedding_irrep,
    avg_neigh=1,
    nonlinearity_type="gate",
    resnet=False,
    nonlinearity_scalars={"e": "silu", "o": "tanh"},
    nonlinearity_gates={"e": "silu", "o": "abs"},
    radial_network_hidden_dim=64,
    radial_network_layers=2,
) -> torch.nn.Module:
    """
    Get NequIP convolution layer.

    Args:
        parity (bool): whether to use parity
        lmax (int): maximum l value for spherical harmonics
        conv_feature_size (int): convolution feature size
        node_embedding_irrep_in: input node embedding irreps
        node_attr_irrep: node attribute irreps
        edge_attr_irrep: edge attribute irreps
        edge_embedding_irrep: edge embedding irreps
        avg_neigh (int): average number of neighbors <optional, def: 1>
        nonlinearity_type (str): nonlinearity type <optional, def: gate>
        resnet (bool): whether to use resnet <optional, def: False>
        nonlinearity_scalars (dict): nonlinearity scalars <optional, def: {e: ssp, o: tanh}>
        nonlinearity_gates (dict): nonlinearity gates <optional, def: {e: ssp, o: abs}>
        radial_network_hidden_dim (int): radial network hidden dimension <optional, def: 64>
        radial_network_layers (int): radial network layers <optional, def: 2>

    Returns:
        torch.nn.Module: NequIP convolution layer
    """
    from ..layers._convnetlayer import ConvNetLayer

    conv_hidden_irrep = Irreps(
        [
            (conv_feature_size, (l, p))
            for p in ((1, -1) if parity else (1,))
            for l in range(lmax + 1)
        ]
    )
    conv_kw = {
        "invariant_layers": radial_network_layers,
        "invariant_neurons": radial_network_hidden_dim,
        "avg_num_neighbors": avg_neigh,
    }

    conv_layer = ConvNetLayer(
        node_embedding_irrep_in,
        conv_hidden_irrep,
        node_attr_irrep,
        edge_attr_irrep,
        edge_embedding_irrep,
        convolution_kwargs=conv_kw,
        resnet=resnet,
        nonlinearity_type=nonlinearity_type,
        nonlinearity_scalars=nonlinearity_scalars,
        nonlinearity_gates=nonlinearity_gates,
    )
    return conv_layer


def get_nequip_conv_block(
    n_conv_layers: int,
    parity: bool,
    lmax: int,
    conv_feature_size: int,
    node_embedding_irrep_in,
    node_attr_irrep,
    edge_attr_irrep,
    edge_embedding_irrep,
    avg_neigh=1,
    nonlinearity_type="gate",
    resnet=False,
    nonlinearity_scalars: dict = {"e": "silu", "o": "tanh"},
    nonlinearity_gates: dict = {"e": "silu", "o": "abs"},
    radial_network_hidden_dim=64,
    radial_network_layers=2,
    graph_type="mic",  # "mic/staged"
) -> torch.nn.Module:
    """
    Returns NequIP convolution block, with multiple convolution layers.

    Args:
        n_conv_layers (int): number of conv layers
        parity (bool): whether to use parity
        lmax (int): maximum l value for spherical harmonics
        conv_feature_size (int): convolution feature size
        node_embedding_irrep_in: input node embedding irreps
        node_attr_irrep: node attribute irreps
        edge_attr_irrep: edge attribute irreps
        edge_embedding_irrep: edge embedding irreps
        avg_neigh (int): average number of neighbors <optional, def: 1>
        nonlinearity_type (str): nonlinearity type <optional, def: gate>
        resnet (bool): whether to use resnet <optional, def: False>
        nonlinearity_scalars (dict): nonlinearity scalars <optional, def: {e: ssp, o: tanh}>
        nonlinearity_gates (dict): nonlinearity gates <optional, def: {e: ssp, o: abs}>
        radial_network_hidden_dim (int): radial network hidden dimension <optional, def: 64>
        radial_network_layers (int): radial network layers <optional, def: 2>
        graph_type (str): type of graph <optional, def: staged>

    Returns:
        torch.nn.Module: NequIP convolution block
    """
    layers = []
    last_node_irrep = node_embedding_irrep_in
    for i in range(n_conv_layers):
        conv_layer = get_nequip_conv(
            parity=parity,
            lmax=lmax,
            conv_feature_size=conv_feature_size,
            node_embedding_irrep_in=last_node_irrep,
            node_attr_irrep=node_attr_irrep,
            edge_attr_irrep=edge_attr_irrep,
            edge_embedding_irrep=edge_embedding_irrep,
            avg_neigh=avg_neigh,
            nonlinearity_type=nonlinearity_type,
            resnet=resnet,
            nonlinearity_scalars=nonlinearity_scalars,
            nonlinearity_gates=nonlinearity_gates,
            radial_network_hidden_dim=radial_network_hidden_dim,
            radial_network_layers=radial_network_layers,
        )
        layers.append(conv_layer)
        last_node_irrep = conv_layer.irreps_out

    if graph_type == "mic":
        nequip_model_str = re.sub(
            r"<<edge_length_embeddings>>", "edge_length_embeddings,", NEQUIP_BLOCK
        )
        nequip_model_str = re.sub(r"<<edge_sh>>", "edge_sh,", nequip_model_str)
        nequip_model_str = re.sub(r"<<edge_index>>", "edge_index,", nequip_model_str)
        conv_block = """
        for layer in self.conv_layers:
            h = layer(x, h, edge_length_embeddings, edge_sh, edge_index)
        """
        nequip_model_str = re.sub(r"<<CONV>>", conv_block, nequip_model_str)

    elif graph_type == "staged":
        edge_length_embedding = ""
        edge_sh = ""
        edge_index = ""

        for layer in range(n_conv_layers):
            edge_length_embedding += f"edge_length_embeddings_{layer},"
            edge_sh += f"edge_sh_{layer},"
            edge_index += f"edge_index_{layer},"

        nequip_model_str = re.sub(
            r"<<edge_length_embeddings>>", edge_length_embedding, NEQUIP_BLOCK
        )
        nequip_model_str = re.sub(r"<<edge_sh>>", edge_sh, nequip_model_str)
        nequip_model_str = re.sub(r"<<edge_index>>", edge_index, nequip_model_str)
        conv_block = ""

        for i, i_inv in zip(range(n_conv_layers), range(n_conv_layers - 1, -1, -1)):
            conv_block += f"""
        h = self.conv_layers[{i}](x, h, edge_length_embeddings_{i_inv}, edge_sh_{i_inv}, edge_index_{i_inv})"""
        nequip_model_str = re.sub(r"<<CONV>>", conv_block, nequip_model_str)

    else:
        raise ValueError(f"Unknown graph type {graph_type}")

    local_scope = {}
    exec(nequip_model_str, globals(), local_scope)
    NequipConvBlock = local_scope["NequipConvBlock"]
    conv_block = NequipConvBlock(
        n_layers=n_conv_layers,
        conv_layers=layers,
    )

    return conv_block
