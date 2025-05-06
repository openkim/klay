from enum import Enum


class Layers(Enum):
    """
    Supported layers types
    """

    ELEM_EMBEDDING = 0
    EDGE_EMBEDDING = 1
    RADIAL_BASIS = 2
    LINEAR_E3NN = 3
    NEQUIP_CONV = 4
    NEQUIP_CONV_BLOCK = 5
    EGNN_CONV = 5
    TORCH_NN = 6
    TORCH_FUNC = 7


class ElemEmbedding(Enum):
    """
    Supported element embedding types.
    1. ONE_HOT: One hot encoding, with n_elems as number of elements
    2. BINARY: Binary encoding, 8 bits for atomic number, no n_elems required
    3. ELECTRON: Electron state encoding, no n_elems required. length of repr : 24
    """

    ONE_HOT = "one_hot"
    BINARY = "binary"
    ELECTRON = "electron"
