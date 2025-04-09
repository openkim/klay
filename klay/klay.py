from typing import Union
import torch
from e3nn.o3 import Irreps
from e3nn.util import jit
from .layers import embedding as e
from .layers import AtomwiseLinear
from .layers.egnn import EGCL

import yaml
import sys


def summary():
    """
    Print summary of supported layers, and their parameters required.
    """
    print("YAML blocks heading:")
    print(" - elem_embedding: for getting element embeddings")
    print("   - embedding_type: one_hot, binary, electron")
    print("   - n_elems: number of elements <only for one_hot>")
    print(" - edge_embedding: for getting edge embeddings")
    print("   - lmax: maximum l value for spherical harmonics")
    print(
        "   - normalize: whether to normalize the spherical harmonics <optional, def: True>"
    )
    print(
        "   - normalization: normalization scheme to use <optional, def: component>"
    )
    print("   - parity: whether to use parity <optional, def: True>")
    print(" - radial_basis: for getting radial basis")
    print("   - r_max: cutoff radius")
    print("   - num_basis: number of basis functions <optional, def: 8>")
    print(
        "   - trainable: whether the basis functions are trainable <optional, def: True>"
    )
    print("   - power: power used in envelope function <optional, def: 6>")
    print(" - linear_e3nn: for getting linear e3nn")
    print("   - irreps_in: input irreps")
    print("   - irreps_out: output irreps")
    print(" - nequip_conv: for getting nequip conv")
    print("   - parity: whether to use parity")
    print("   - lmax: maximum l value for spherical harmonics")
    print("   - conv_feature_size: convolution feature size")
    print("   - node_embedding_irrep_in: input node embedding irreps")
    print("   - node_attr_irrep: node attribute irreps")
    print("   - edge_attr_irrep: edge attribute irreps")
    print("   - edge_embedding_irrep: edge embedding irreps")
    print("   - avg_neigh: average number of neighbors <optional, def: 1>")
    print("   - nonlinearity_type: nonlinearity type <optional, def: gate>")
    print("   - resnet: whether to use resnet <optional, def: False>")
    print(
        "   - nonlinearity_scalars: nonlinearity scalars <optional, def: {e: ssp, o: tanh}>"
    )
    print(
        "   - nonlinearity_gates: nonlinearity gates <optional, def: {e: ssp, o: abs}>"
    )
    print(
        "   - radial_network_hidden_dim: radial network hidden dimension <optional, def: 64>"
    )
    print(
        "   - radial_network_layers: radial network layers <optional, def: 2>"
    )
    print(" -------------------------------------------------------------")
    print(" Better is to use the NequIPConvBlock for multiple conv layers")
    print(" - nequip_conv_block: for getting nequip conv block")
    print("   - n_conv_layers: number of conv layers")
    print("   - parity: whether to use parity")
    print("   - lmax: maximum l value for spherical harmonics")
    print("   - conv_feature_size: convolution feature size")
    print("   - node_embedding_irrep_in: input node embedding irreps")
    print("   - node_attr_irrep: node attribute irreps")
    print("   - edge_attr_irrep: edge attribute irreps")
    print("   - edge_embedding_irrep: edge embedding irreps")


def get_element_embedding(
    embedding_type: str, n_elems: int = 118
) -> torch.nn.Module:
    """
    Get torch module for element embedding.

    Args:
        embedding_type (str): element embedding type
        n_elems (int): number of elements <only for one_hot>

    Returns:
        torch.nn.Module: element embedding module
    """
    pass
    # embedding_type_enum = ElemEmbedding.get_embed_type_from_str(embedding_type)
    # if embedding_type_enum == ElemEmbedding.ONE_HOT:
    #     return e.OneHotAtomEncoding(n_elems)
    # elif embedding_type_enum == ElemEmbedding.BINARY:
    #     return e.BinaryAtomicNumberEncoding()
    # elif embedding_type_enum == ElemEmbedding.ELECTRON:
    #     return e.ElectronicConfigurationEncoding()
    # else:
    #     raise ValueError(f"Unknown element embedding type: {embedding_type}")


def get_edge_embedding(
    lmax: int,
    normalize: bool = True,
    normalization: str = "component",
    parity: bool = True,
) -> torch.nn.Module:
    """
    Returns edge embedding module. Edge embedding return spherical harmonics for the edge vectors with total length being (lmax + 1) * (lmax + 1).

    Args:
        lmax (int): maximum l value for spherical harmonics
        normalize (bool): whether to normalize the spherical harmonics <optional, def: True>
        normalization (str): normalization scheme to use <optional, def: component>
        parity (bool): whether to use parity <optional, def: True>

    Returns:
        torch.nn.Module: edge embedding module
    """
    if parity:
        p = -1
    else:
        p = 1
    irreps = Irreps([(1, (l, p**l)) for l in range(lmax + 1)])
    return e.SphericalHarmonicEdgeAttrs(
        irreps, edge_sh_normalize=normalize, edge_sh_normalization=normalization
    )


def get_radial_basis(
    r_max: Union[float, torch.Tensor],
    num_basis: int = 8,
    trainable: bool = True,
    power: int = 6,
) -> torch.nn.Module:
    """
    Returns radial basis module. Radial basis module returns the radial Bessel basis functions for the edge lengths.

    Args:
        r_max (Union[float, torch.Tensor]): cutoff radius
        num_basis (int): number of basis functions <optional, def: 8>
        trainable (bool): whether the basis functions are trainable <optional, def: True>
        power (int): power used in envelope function <optional, def: 6>

    Returns:
        torch.nn.Module: radial basis module
    """
    basis_kwargs = {
        "r_max": r_max,
        "num_basis": num_basis,
        "trainable": trainable,
    }
    cutoff_kwargs = {"r_max": r_max, "p": power}
    return e.RadialBasisEdgeEncoding(
        basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs
    )


def get_linear_e3nn(irreps_in, irreps_out) -> torch.nn.Module:
    """
    Get linear e3nn module.

    Args:
        irreps_in: input irreps
        irreps_out: output irreps

    Returns:
        torch.nn.Module: linear e3nn module
    """
    return AtomwiseLinear(irreps_in, irreps_out)



def get_egnn_conv(in_node_fl, hidden_node_fl, edge_fl=0, act_fn=torch.nn.SiLU(), n_hidden_layers=1, normalize_radial=False):
    return EGCL(in_node_fl, hidden_node_fl, edge_fl, act_fn, n_hidden_layers, normalize_radial)


def get_torch_nn_layer(layer_type, layer_params):
    return getattr(torch.nn, layer_type)(**layer_params)


def get_torch_func_layer(func_name):
    return getattr(torch.nn.functional, func_name)


def get_model_layers_from_yaml(yaml_file):
    """
    Generate model sequentially from yaml file. Ordering in YAML file is important. Order of layers is important.
    Under the model block, the layers are defined as a list of dictionaries. Each dictionary contains the layer type and its parameters.

    Here the expected inputs to the model are:
    1. atomic numbers: torch.int64 tensor of shape (n_atoms,)
    2. positions: default tensor type of shape (n_atoms, 3)
    3. edge_indices: torch.int64 tensor of shape (n_edges, 2)
    4. shifts: default tensor type of shape (n_edges, 3) (actual vectors = positions[edge_indices[:, 1]] - positions[edge_indices[:, 0]] - shifts)

    Example yaml file:
    model:
        - elem_embedding:
            embedding_type: one_hot
            n_elems: 118

        - edge_embedding:
            lmax: 6
            normalize: True
            normalization: component
            parity: True

        - radial_basis:
            r_max: 5.0
            num_basis: 8
            trainable: True
            power: 6

        - linear_e3nn:
            irreps_in: DETECT_PREV
            irreps_out: 32x0e

        - nequip_conv_block:
            n_conv_layers: 3
            parity: True
            lmax: 6
            conv_feature_size: 64
            node_embedding_irrep_in: DETECT_PREV
            node_attr_irrep: DETECT_PREV
            edge_attr_irrep: DETECT_PREV
            edge_embedding_irrep: DETECT_PREV
            avg_neigh: 1
            resnet: True
            radial_network_hidden_dim: 64
            radial_network_layers: 2

        - linear_e3nn:
            irreps_in: DETECT_PREV
            irreps_out: 1x0e

    Args:
        yaml_file: path to yaml file

    Returns:
        torch.nn.Sequential: model
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    layers_list = config["model"]

    layers = []
    node_attr_irrep = None
    edge_attr_irrep = None
    edge_length_embedding_irrep = None

    for layer_dict in layers_list:
        for layer_type, layer_params in layer_dict.items():
            if layer_type == "elem_embedding":
                elem_embed = layer_params
                layers.append(get_element_embedding(**elem_embed))
                elem_embed["irreps_out"] = layers[-1].irreps_out
                node_attr_irrep = layers[-1].irreps_out
            elif layer_type == "edge_embedding":
                edge_embed = layer_params
                layers.append(get_edge_embedding(**edge_embed))
                edge_embed["irreps_out"] = layers[-1].irreps_out
                edge_attr_irrep = layers[-1].irreps_out
            elif layer_type == "radial_basis":
                radial_basis = layer_params
                layers.append(get_radial_basis(**radial_basis))
                radial_basis["irreps_out"] = layers[-1].irreps_out
                edge_length_embedding_irrep = layers[-1].irreps_out
            elif layer_type == "linear_e3nn":
                if layer_params["irreps_in"] == "DETECT_PREV":
                    layer_params["irreps_in"] = layers[-1].irreps_out
                linear_e3nn = layer_params
                layers.append(get_linear_e3nn(**linear_e3nn))
            elif layer_type == "nequip_conv_block":
                nequip_conv_block = layer_params
                if (
                    nequip_conv_block["node_embedding_irrep_in"]
                    == "DETECT_PREV"
                ):
                    nequip_conv_block["node_embedding_irrep_in"] = layers[
                        -1
                    ].irreps_out
                if nequip_conv_block["node_attr_irrep"] == "DETECT_PREV":
                    nequip_conv_block["node_attr_irrep"] = node_attr_irrep
                if nequip_conv_block["edge_embedding_irrep"] == "DETECT_PREV":
                    nequip_conv_block[
                        "edge_embedding_irrep"
                    ] = edge_length_embedding_irrep
                if nequip_conv_block["edge_attr_irrep"] == "DETECT_PREV":
                    nequip_conv_block["edge_attr_irrep"] = edge_attr_irrep
                # layers.append(get_nequip_conv_block(**nequip_conv_block))

            elif layer_type == "egnn_conv":
                egnn_conv = layer_params
                layers.append(get_egnn_conv(**egnn_conv))
            elif layer_type == "torch_nn":
                torch_nn = layer_params["name"]
                params = layer_params["kwargs"]

                layers.append(get_torch_nn_layer(torch_nn, params))
            elif layer_type == "torch_func":
                torch_func = layer_params["name"]
                layers.append(get_torch_func_layer(torch_func))

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")


    trainable_parameters = 0
    for layer in layers:
        try:
            trainable_parameters += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        except AttributeError:
            pass

    print("---------------------------------------------------------")
    print(f"Generated layers with {trainable_parameters} parameters")
    print("---------------------------------------------------------")
    return layers
