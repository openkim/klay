from typing import Any, Callable, List, Optional

import torch
from e3nn import o3
from omegaconf import DictConfig

from ...core import ModuleCategory, register
from ...utils import irreps_blocks_to_string
from .._base import _BaseLayer
from .blocks import EquivariantProductBasisBlock, RealAgnosticResidualInteractionBlock


@register(
    "MACE_layer",
    category=ModuleCategory.CONVOLUTION,
    inputs=["vectors", "node_feats", "node_attrs", "edge_feats", "edge_index"],
    outputs=["node_feats"],
)
class MACE_layer(_BaseLayer, torch.nn.Module):
    r"""A MACE layer from the `"MACE: Higher Order Equivariant Message Passing Neural Networks
    for Fast and Accurate Force Fields, Neurips 2022"
    <https://arxiv.org/abs/2206.07697>`_ paper
    Construct a single layer of the MACE architecture for efficient higher order equivariant message
    passing.

    Args:
        max_ell (int): Maximum angular momentum in the spherical expansion on edges, :math:`l = 0, 1, \dots`.
        Controls the resolution of the spherical expansion.
        correlation (int): The maximum correlation order of the messages, :math:`\nu = 0, 1, \dots`.
        n_dims_in (int): The number of input node attributes.
        hidden_irreps (str): The hidden irreps defining the node features to construct.
        node_feats_irreps (str): The irreps of the node features in the input.
        edge_feats_irreps (str): The irreps of the edge features in the input.
        avg_num_neighbors (float): A normalization factor for the pooling operation,
        usually taken as the average number of neighbors.
        interaction_cls (Callable, optional): The type of interaction block to use.
        Defaults to RealAgnosticResidualInteractionBlock.
        element_dependent (bool, optional): Whether to use element dependent basis functions.
        Defaults to False.
        use_sc (bool, optional): Whether to use the self connection. Defaults to True.

    Shapes:
        - **input:**
            - **vectors** (torch.Tensor): The edge vectors of shape :math:`(|\mathcal{E}|, 3)`.
            - **node_feats** (torch.Tensor): The node features of shape :math:`(|\mathcal{V}|, \text{node\_feats\_irreps})`.
            - **node_attrs** (torch.Tensor): The node attributes of shape :math:`(|\mathcal{V}|, \text{n\_dims\_in})`.
            - **edge_feats** (torch.Tensor): The edge features of shape :math:`(|\mathcal{E}|, (\text{egde\_feats\_irreps}))`.
            - **edge_index** (torch.Tensor): The edge indices of shape :math:`(2, |\mathcal{E}|)`.
        - **output:**
            - **node_feats** (torch.Tensor): The node features of shape :math:`(|\mathcal{V}|, \text{hidden\_irreps})`.
    """

    def __init__(
        self,
        max_ell: int,
        correlation: int,
        n_dims_in: int,
        hidden_irreps: str,
        node_feats_irreps: str,
        edge_feats_irreps: str,
        avg_num_neighbors: float,
        interaction_cls: Optional[Callable] = RealAgnosticResidualInteractionBlock,
        use_sc: bool = True,
        node_attr_irreps: Optional[str] = None,
    ):
        super().__init__()
        if node_attr_irreps is None:
            node_attr_irreps = o3.Irreps([(n_dims_in, (0, 1))])
        else:
            node_attr_irreps = o3.Irreps(node_attr_irreps)
        hidden_irreps = o3.Irreps(hidden_irreps)
        node_feats_irreps = o3.Irreps(node_feats_irreps)
        edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        self.interaction = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.interaction.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=n_dims_in,
            use_sc=use_sc,
        )

    def forward(
        self,
        vectors: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_attrs = self.spherical_harmonics(vectors)
        node_feats, sc = self.interaction(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        node_feats = self.product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
        return node_feats

    @classmethod
    def from_config(
        cls,
        lmax: int,
        correlation: int,
        num_elements: int,
        hidden_irreps_block: List[dict[str, Any]],
        input_block: List[dict[str, Any]],
        num_radial_basis: int = 8,
        node_attr_block=None,
        avg_num_neighbors=1.0,
        use_sc=True,
    ) -> "MACE_layer":
        """Create a new instance from the config.

        Args:

            lmax (int): maximum l for spherical harmonics
            correlation (int): The maximum correlation order of the messages, :math:`\nu = 0, 1, \dots`.
            num_elements (int): The number of input node attributes.
            hidden_irreps_block (List[dict[str, Any]]): The irreps of the hidden features in the input.
            input_block (List[dict[str, Any]]): The irreps of the node features in the input.
            num_radial_basis (int): number of radial basis functions
            node_attr_block (List[dict[str, Any]]): The irreps of the node attributes in the input, usually nelements x 0e.
            avg_num_neighbors (float): A normalization factor for the pooling operation,
            usually taken as the average number of neighbors.
            use_sc(bool): Whether to use the self connection. Defaults to True.
        """
        max_ell = lmax
        hidden_irreps = irreps_blocks_to_string(hidden_irreps_block)
        node_feats_irreps = irreps_blocks_to_string(input_block)
        edge_feats_irreps = f"{num_radial_basis}x0e"
        if node_attr_block is None:
            # default is one hot like encoding
            node_attr_irreps = f"{num_elements}x0e"
        else:
            node_attr_irreps = irreps_blocks_to_string(node_attr_block)

        return cls(
            max_ell=max_ell,
            correlation=correlation,
            n_dims_in=num_elements,
            hidden_irreps=hidden_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_feats_irreps=edge_feats_irreps,
            avg_num_neighbors=avg_num_neighbors,
            node_attr_irreps=node_attr_irreps,
            use_sc=use_sc,
        )
