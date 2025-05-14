from enum import Enum, auto


class ModuleCategory(Enum):
    """Enum for different module categories in a neural network. Register new categories
    here if needed."""

    EMBEDDING = auto()
    CONVOLUTION = auto()
    LINEAR = auto()
    ATTENTION = auto()
    AUTOGRAD = auto()
    ARBITRARY = auto()
