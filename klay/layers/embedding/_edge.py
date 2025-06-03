from typing import Optional, Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from ...core import ModuleCategory, register
from .._base import _BaseLayer
from ..cutoffs import PolynomialCutoff
from ..radial_basis import BesselBasis


@register(
    "SphericalHarmonicEdgeAttrs",
    inputs=["pos", "edge_index", "shift"],
    outputs=["edge_vec", "edge_lengths", "edge_sh"],
    category=ModuleCategory.EMBEDDING,
)
@compile_mode("script")
class SphericalHarmonicEdgeAttrs(_BaseLayer, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    def __init__(
        self,
        irreps_in: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
    ):
        super().__init__()

        if isinstance(irreps_in, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_in)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_in)
        self.irreps_in = irreps_in
        self.irreps_out = self.irreps_edge_sh
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, pos, edge_index, shift: Optional[torch.Tensor] = None):
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]  # 0 -> 1
        if shift is None:
            shift = edge_vec.new_zeros(edge_vec.shape)
        edge_vec = edge_vec + shift
        edge_lengths = torch.linalg.norm(edge_vec, dim=1)
        edge_sh = self.sh(edge_vec)
        return edge_vec, edge_lengths, edge_sh

    @classmethod
    def from_config(cls, *, lmax: int = 1, normalization: Optional[str] = "component"):
        """Create a new instance from the config.

        Args:
            lmax (int): maximum l for spherical harmonics
            normalization (str): normalization scheme to use
        """
        edge_sh_normalization = normalization is not None
        return cls(
            irreps_in=lmax,
            edge_sh_normalization=normalization,
            edge_sh_normalize=edge_sh_normalization,
        )


@register(
    "RadialBasisEdgeEncoding",
    inputs=["edge_lengths"],
    outputs=["edge_length_embedded"],
    category=ModuleCategory.EMBEDDING,
)
@compile_mode("script")
class RadialBasisEdgeEncoding(_BaseLayer, torch.nn.Module):
    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.irreps_out = o3.Irreps([(self.basis.num_basis, (0, 1))])

    def forward(self, edge_lengths):
        edge_length_embedded = self.basis(edge_lengths) * self.cutoff(edge_lengths)[:, None]
        return edge_length_embedded

    @classmethod
    def from_config(
        cls,
        r_max: float,
        num_radial_basis: int = 8,
        polynomial_degree: float = 6,
        radial_basis_trainable: bool = True,
        basis: str = "BesselBasis",
        cutoff: str = "PolynomialCutoff",
        basis_kwargs={},
        cutoff_kwargs={},
    ):
        """Create a new instance from the config.

        Args:
            r_max (float): cutoff radius
            num_radial_basis (int): number of radial basis functions
            polynomial_degree (float): degree of the polynomial cutoff
            radial_basis_trainable (bool): whether the radial basis function is trainable
            basis (str): name of the radial basis function
            cutoff (str): name of the cutoff function
            basis_kwargs (dict): kwargs for the radial basis function
            cutoff_kwargs (dict): kwargs for the cutoff function
        """
        if basis == "BesselBasis":
            basis = BesselBasis
        else:
            raise ValueError(f"Unknown basis: {basis}")

        basis_kwargs |= {
            "r_max": r_max,
            "num_basis": num_radial_basis,
            "trainable": radial_basis_trainable,
        }

        if cutoff == "PolynomialCutoff":
            cutoff = PolynomialCutoff
        else:
            raise ValueError(f"Unknown cutoff: {cutoff}")

        cutoff_kwargs |= {
            "r_max": r_max,
            "p": polynomial_degree,
        }

        return cls(
            basis=basis,
            cutoff=cutoff,
            basis_kwargs=basis_kwargs,
            cutoff_kwargs=cutoff_kwargs,
        )
