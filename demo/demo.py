import ase.io
from ocpmodels.preprocessing import AtomsToGraphs
import torch
from simple_nequip import gen_model
from e3nn.util import jit


data = ase.io.read("./demo/Si4.xyz",":")
graph_gen = AtomsToGraphs(radius=4.0)
graph_data = graph_gen.convert_all(data)

gd = graph_data[0] # one instance of graph data

x = torch.zeros_like(gd.atomic_numbers, dtype=torch.int64)
model, layer_dict = gen_model("in.yaml")
model = jit.script(model)
# model = torch.jit.load("Si_nequip_EF.pt")
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params: {pytorch_total_params}")

gd.pos.requires_grad_(True)

E_atomwise,F_nonconservative = model(x, gd.pos, gd.edge_index, gd.cell, gd.cell_offsets)
F_conservative, = torch.autograd.grad([E_atomwise.sum()], [gd.pos])
# print(E_atomwise,F_nonconservative, F_conservative)