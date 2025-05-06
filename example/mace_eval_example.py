import torch

torch.set_default_dtype(torch.float64)  # for reproducibility

from klay.builder import build_model
from klay.io import load_config

# 1. load & build
cfg = load_config("./mace_model.yaml")  # path to *your* YAML
model = build_model(cfg).eval()  # no gradients needed

# 2. pick a toy system size
N = 6  # number of atoms
E = 2 * N  # edges (rough heuristic)

# 3. craft dummy inputs matching `model_inputs`
batch = {
    # ---- atomic numbers (ints 1–10)
    "atomic_numbers": torch.zeros(N, dtype=torch.long),
    # ---- positions (Å) – random cube of side 5 Å
    "positions": torch.rand(N, 3, dtype=torch.float64, requires_grad=True) * 5.0,
    # ---- edge list – random undirected graph, self-loops removed
    "edge_index": torch.randint(0, N, (2, E), dtype=torch.long),
}

# 4. forward pass
print(model)
out = model(**batch)

print(out)
# print({k: v.shape for k, v in out.items()})
# e.g. {'energy': torch.Size([]), 'representation': torch.Size([N, 32])}
