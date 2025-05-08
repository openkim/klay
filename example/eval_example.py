import torch

torch.set_default_dtype(torch.float64)  # for reproducibility

from klay.builder.fx_builder import build_fx_model
from klay.io.load_config import load_config

# ---------------------------------------------------------------------
# 1. load & build
cfg = load_config("new_model.yaml")  # path to *your* YAML
model = build_fx_model(cfg)  # no gradients needed

from e3nn.util import jit

model = jit.script(model)  # optional, for faster inference

# ---------------------------------------------------------------------
# 2. pick a toy system size
N = 6  # number of atoms
E = 2 * N  # edges (rough heuristic)

# ---------------------------------------------------------------------
# 3. craft dummy inputs matching `model_inputs`
batch = {
    # ---- atomic numbers (ints 1–10)
    "atomic_numbers": torch.zeros(N, dtype=torch.long),
    # ---- positions (Å) – random cube of side 5 Å
    "positions": torch.rand(N, 3) * 5.0,
    # ---- edge list – random undirected graph, self-loops removed
    "edge_index": torch.randint(0, N, (2, E), dtype=torch.long),
}

# your model may expect a `shifts` tensor (for PBC) or others:
if "shifts" in cfg.get("model_inputs", {}):
    batch["shifts"] = torch.zeros(E, 3)  # no periodic offsets

# ---------------------------------------------------------------------
# 4. forward pass
print(model)
with torch.no_grad():
    out = model(**batch)

print(out)
# print({k: v.shape for k, v in out.items()})
# e.g. {'energy': torch.Size([]), 'representation': torch.Size([N, 32])}
