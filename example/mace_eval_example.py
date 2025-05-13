import torch

torch.set_default_dtype(torch.float64)  # for reproducibility

from e3nn.util import jit

from klay.builder import build_model
from klay.io import load_config

# 1. load & build
cfg = load_config("./mace_model.yaml")  # path to *your* YAML
model = build_model(cfg)  # no gradients needed

# model = jit.script(model)  # optional, but recommended for performance

# 2. pick a toy system size
N = 6  # number of atoms
E = 2 * N  # edges (rough heuristic)

# 3. craft dummy inputs matching `model_inputs`
batch = {
    # ---- atomic numbers (ints 1–10)
    "species": torch.zeros(N, dtype=torch.long),
    # ---- positions (Å) – random cube of side 5 Å
    "coords": torch.rand(N, 3, dtype=torch.float64, requires_grad=True) * 5.0,
    # ---- edge list – random undirected graph, self-loops removed
    "edge_index0": torch.randint(0, N, (2, E), dtype=torch.long),
    # ---- batch – batch indices for each node"
    "contributing": torch.zeros(N, dtype=torch.long),
}

batch["contributing"][N // 3 :] = 1  # two batches of size N//3 and N//3+N%3
batch["contributing"][2 * N // 3 :] = 2  # two batches of size N//3 and N//3+N%3

# 4. forward pass
print("In training mode")
out = model(**batch)
print(out)

model = model.eval()  # switch to eval mode
print("In eval mode")
out = model(**batch)  # no gradients needed
print(out)
# print({k: v.shape for k, v in out.items()})
# e.g. {'energy': torch.Size([]), 'representation': torch.Size([N, 32])}
