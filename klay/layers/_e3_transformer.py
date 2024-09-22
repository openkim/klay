import torch
from e3nn import o3


class E3Transformer(torch.nn.Module):
    """
    E(3)-equivariant transformer
    https://docs.e3nn.org/en/latest/guide/transformer.html
    """

    def __init__(self, irrep_input, irrep_output, irrep_query, irrep_key):
        super().__init__()

        self.irrep_input = irrep_input
        self.irrep_output = irrep_output
        self.irrep_query = irrep_query
        self.irrep_key = irrep_key

        self.h_q = o3.Linear(irrep_input, irrep_query)
        self.h_k = o3.Linear(irrep_input, irrep_key)

        tp_k = o3.FullyConnectedTensorProduct(irrep_input, irrep_sh, irrep_key, shared_weights=False)
        fc_k = nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel], act=torch.nn.functional.silu)
        tp_v = o3.FullyConnectedTensorProduct(irrep_input, irrep_sh, irrep_output, shared_weights=False)
        fc_v = nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=torch.nn.functional.silu)

