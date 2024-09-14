import torch
from e3nn.o3 import Irreps
from e3nn.util import jit
from layers import embedding as e
from layers import AtomwiseLinear
from layers import ConvNetLayer, InteractionBlock

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
    # EGNN_CONV = 5

class ElemEmbedding(Enum):
    """
    """
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
    """
    """
    embedding_type_enum = ElemEmbedding.get_embed_type_from_str(embedding_type)
    if embedding_type_enum == ElemEmbedding.ONE_HOT:
        return e.OneHotAtomEncoding(n_elems)
    elif embedding_type_enum == ElemEmbedding.BINARY:
        return e.BinaryAtomEncoding()
    elif embedding_type_enum == ElemEmbedding.ELECTRON:
        return e.ElectronAtomEncoding()
    else:
        raise ValueError(f"Unknown element embedding type: {embedding_type}")


def get_edge_embedding(lmax, normalize=True, normalization="component", parity=True):
    """
    """
    if parity:
        p = -1
    else:
        p = 1
    irreps = Irreps([(1, (l, p**l)) for l in range(lmax + 1)])
    return e.SphericalHarmonicEdgeAttrs(irreps, edge_sh_normalize=normalize, edge_sh_normalization=normalization)

def get_radial_basis(r_max, num_basis=8, trainable=True, power=6):
    """
    """
    basis_kwargs = {"r_max": r_max, "num_basis": num_basis, "trainable": trainable}
    cutoff_kwargs = {"r_max": r_max, "p": power}
    return e.RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs)
    

def get_linear_e3nn(irreps_in, irreps_out):
    """
    """
    return AtomwiseLinear(irreps_in, irreps_out)

def get_nequip_conv(node_embedding_irrep_in, feature_irreps_hidden, node_attr_irrep, edge_attr_irrep, edge_embedding_irrep, nonlinearity_type="gate", resnet=False, nonlinearity_scalars = {"e": "ssp", "o": "tanh"}, nonlinearity_gates = {"e": "ssp", "o": "abs"}):
    """
    """
    
    

def gen_model(yaml_file, save=True):

    # atomwise embedding
    
    # convolution layers
    node_feature_irrep_intermidiate = []
    conv_hidden_irrep = Irreps([(config["conv_feature_size"],(l,p)) for p in ((1, -1) if config["parity"] else (1,)) for l in range(config["lmax"] + 1)])
    invariant_neurons = config["radial_network_hidden_dim"]
    invariant_layers = config["radial_network_layers"]
    average_num_neigh = config["average_num_neigh"]
    conv_kw = {"invariant_layers": invariant_layers, "invariant_neurons": invariant_neurons, "avg_num_neighbors": average_num_neigh}

    last_node_irrep = node_feature_irrep

    for i in range(config["n_conv_layers"]):
        conv_layer = ConvNetLayer(last_node_irrep,
                conv_hidden_irrep,
                node_attr_irrep,
                edge_attr_irrep,
                edge_feature_irrep,
                convolution_kwargs=conv_kw)
        node_embedding_dim_list.append(conv_layer.irreps_out)
        layer_dict[f"conv:{i}"] = conv_layer
        last_node_irrep = conv_layer.irreps_out

    # final mappings
    for i in range(config["post_conv_layers"] - 1):
        linear_post_layer = AtomwiseLinear(last_node_irrep, config["post_conv_irreps"][i])
        layer_dict[f"post_conv:{i}"] = linear_post_layer
        last_node_irrep = config["post_conv_irreps"][i]
        node_embedding_dim_list.append(last_node_irrep)

    final_conv_irreps = "{}x0e".format(config["num_targets"]) # config["post_conv_irreps"][-1]
    linear_post_layer = AtomwiseLinear(last_node_irrep, final_conv_irreps)
    layer_dict["post_conv:{}".format(config["post_conv_layers"] - 1)] = linear_post_layer
    last_node_irrep = config["post_conv_irreps"][i]
    node_embedding_dim_list.append(last_node_irrep)

    # print(last_node_irrep)
    # layer_dict[f"post_conv_activation"] = load_activation(config["post_conv_activation"])
    # layer_dict[f"post_conv:0"] = torch.nn.Linear(
    #     int(last_node_irrep.split('x')[0]), 
    #     config["post_conv_irreps"][0] * config["num_targets"] # parallelize of output layers
    # )
    # last_node_irrep = config["post_conv_irreps"][0]
    # node_embedding_dim_list.append(last_node_irrep)
    
    # for i in range(1, config["post_conv_layers"]):
    #     linear_post_layer = torch.nn.Linear(last_node_irrep, config["post_conv_irreps"][i])
    #     layer_dict[f"post_conv:{i}"] = linear_post_layer
    #     last_node_irrep = config["post_conv_irreps"][i]
    #     node_embedding_dim_list.append(last_node_irrep)

    model = NL_model(
        config,
        layer_dict, 
        energy_scaling_coeff=config["scale"],
        energy_shifting_coeff=config["shift"]
    )
    print(model)
    if save:
        model = jit.script(model)
        model.save(f"{config['model_name']}.pt")
    return model

