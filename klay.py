import torch
from e3nn.o3 import Irreps
from e3nn.util import jit
from layers import embedding as e
from layers import AtomwiseLinear
from layers import ConvNetLayer, NequipConvBlock

import yaml
import sys


from enum import Enum


class Layers(Enum):
    """
    Supported layers
    """

    ELEM_EMBEDDING = 0
    EDGE_EMBEDDING = 1
    RADIAL_BASIS = 2
    LINEAR_E3NN = 3
    NEQUIP_CONV = 4
    NEQUIP_CONV_BLOCK = 5
    # EGNN_CONV = 5        

def get_layers_block():
    print("YAML blocks heading:")
    print(" - elem_embedding: for getting element embeddings")
    print("   - embedding_type: one_hot, binary, electron")
    print("   - n_elems: number of elements <only for one_hot>")
    print(" - edge_embedding: for getting edge embeddings")
    print("   - lmax: maximum l value for spherical harmonics")
    print("   - normalize: whether to normalize the spherical harmonics <optional, def: True>")
    print("   - normalization: normalization scheme to use <optional, def: component>")
    print("   - parity: whether to use parity <optional, def: True>")
    print(" - radial_basis: for getting radial basis")
    print("   - r_max: cutoff radius")
    print("   - num_basis: number of basis functions <optional, def: 8>")
    print("   - trainable: whether the basis functions are trainable <optional, def: True>")
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
    print("   - nonlinearity_scalars: nonlinearity scalars <optional, def: {e: ssp, o: tanh}>")
    print("   - nonlinearity_gates: nonlinearity gates <optional, def: {e: ssp, o: abs}>")
    print("   - radial_network_hidden_dim: radial network hidden dimension <optional, def: 64>")
    print("   - radial_network_layers: radial network layers <optional, def: 2>")
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



class ElemEmbedding(Enum):
    """ """

    ONE_HOT = 0
    BINARY = 1
    ELECTRON = 2

    @staticmethod
    def get_embed_type_from_str(embed_str):
        if embed_str == "one_hot":
            return ElemEmbedding.ONE_HOT
        elif embed_str == "binary":
            return ElemEmbedding.BINARY
        elif embed_str == "electron":
            return ElemEmbedding.ELECTRON
        else:
            raise ValueError(f"Unknown element embedding type: {embed_str}")


def get_element_embedding(embedding_type, n_elems=118):
    """ """
    embedding_type_enum = ElemEmbedding.get_embed_type_from_str(embedding_type)
    if embedding_type_enum == ElemEmbedding.ONE_HOT:
        return e.OneHotAtomEncoding(n_elems)
    elif embedding_type_enum == ElemEmbedding.BINARY:
        return e.BinaryAtomEncoding()
    elif embedding_type_enum == ElemEmbedding.ELECTRON:
        return e.ElectronAtomEncoding()
    else:
        raise ValueError(f"Unknown element embedding type: {embedding_type}")


def get_edge_embedding(
    lmax, normalize=True, normalization="component", parity=True
):
    """ """
    if parity:
        p = -1
    else:
        p = 1
    irreps = Irreps([(1, (l, p**l)) for l in range(lmax + 1)])
    return e.SphericalHarmonicEdgeAttrs(
        irreps, edge_sh_normalize=normalize, edge_sh_normalization=normalization
    )


def get_radial_basis(r_max, num_basis=8, trainable=True, power=6):
    """ """
    basis_kwargs = {
        "r_max": r_max,
        "num_basis": num_basis,
        "trainable": trainable,
    }
    cutoff_kwargs = {"r_max": r_max, "p": power}
    return e.RadialBasisEdgeEncoding(
        basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs
    )


def get_linear_e3nn(irreps_in, irreps_out):
    """ """
    return AtomwiseLinear(irreps_in, irreps_out)


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
    nonlinearity_scalars={"e": "ssp", "o": "tanh"},
    nonlinearity_gates={"e": "ssp", "o": "abs"},
    radial_network_hidden_dim=64,
    radial_network_layers=2,
):
    """ 
    """
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
        nonlinearity_scalars={"e": "ssp", "o": "tanh"},
        nonlinearity_gates={"e": "ssp", "o": "abs"},
        radial_network_hidden_dim=64,
        radial_network_layers=2,
):
    """ """
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
    return NequipConvBlock(n_conv_layers, layers)


def get_model_from_yaml(yaml_file):
    """ 
    Generate model sequentially from yaml file. Ordering in YAML file is important. Order of layers is important.
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

    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    layers_dict = config["model"]
    layers_types = list(layers_dict.keys())

    layers = []
    for layer in layers_types:
        if layer == "elem_embedding":
            elem_embed = layers_dict[layer]
            layers.append(get_element_embedding(**elem_embed))
            elem_embed["irreps_out"] = layers[-1].irreps_out
        elif layer == "edge_embedding":
            edge_embed = layers_dict[layer]
            layers.append(get_edge_embedding(**edge_embed))
            edge_embed["irreps_out"] = layers[-1].irreps_out
        elif layer == "radial_basis":
            radial_basis = layers_dict[layer]
            layers.append(get_radial_basis(**radial_basis))
            radial_basis["irreps_out"] = layers[-1].irreps_out
        elif layer == "linear_e3nn":
            linear_e3nn = layers_dict[layer]
            if linear_e3nn["irreps_in"] == "DETECT_PREV":
                linear_e3nn["irreps_in"] = layers[-1].irreps_out
            layers.append(get_linear_e3nn(**linear_e3nn))
        elif layer == "nequip_conv_block":
            nequip_conv_block = layers_dict[layer]
            if nequip_conv_block["node_embedding_irrep_in"] == "DETECT_PREV":
                nequip_conv_block["node_embedding_irrep_in"] = layers[-1].irreps_out
            if nequip_conv_block["node_attr_irrep"] == "DETECT_PREV":
                nequip_conv_block["node_attr_irrep"] = layers_dict["elem_embedding"]["irreps_out"]
            if nequip_conv_block["edge_embedding_irrep"] == "DETECT_PREV":
                nequip_conv_block["edge_embedding_irrep"] = layers_dict["radial_basis"]["irreps_out"]
            if nequip_conv_block["edge_attr_irrep"] == "DETECT_PREV":
                nequip_conv_block["edge_attr_irrep"] = layers_dict["edge_embedding"]["irreps_out"]
            layers.append(get_nequip_conv_block(**nequip_conv_block))
        else:
            raise ValueError(f"Unknown layer type: {layer}")
        model = torch.nn.Sequential(*layers)
    return model
