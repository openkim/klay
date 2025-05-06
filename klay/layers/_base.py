from abc import ABC, abstractmethod
from typing import Any

import torch


class _BaseLayer(ABC):
    """Base class for all layers in KLAY.

    This class is not intended to be used directly. Instead, use one of the
    subclasses that implement specific layer types.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, **kwargs) -> Any:
        """Create a layer instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary for the layer.

        Returns:
            _BaseMLLayer: An instance of the layer.
        """
        raise NotImplementedError("Subclasses must implement this method.")
