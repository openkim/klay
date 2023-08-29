import ase.io
from ocpmodels.preprocessing import AtomsToGraphs
import torch

model = torch.jit.load("Si_nequip_EF.pt")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params: {pytorch_total_params}")

data = ase.io.read("./Si4.xyz",":")
graph_gen = AtomsToGraphs(radius=4.0)
graph_data = graph_gen.convert_all(data)

gd = graph_data[0] # one instance of graph data

x = torch.zeros_like(gd.atomic_numbers, dtype=torch.int64)

gd.pos.requires_grad_(True)

E_atomwise,F_nonconservative = model(x, gd.pos, gd.edge_index, gd.cell, gd.cell_offsets)
E_total = E_atomwise.sum()
F_conservative, = torch.autograd.grad([E_total], [gd.pos])

print(E_total,F_nonconservative, F_conservative)