import ase.io
from ocpmodels.preprocessing import AtomsToGraphs
import torch
import torch_geometric as pyg

Si_REF = -158.54496821 

# ################## create dataset ################

class SimplePYG(pyg.data.InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = graph_data

#data = ase.io.read("./Si_Cubuck_full_stride_20.xyz",":")
#graph_gen = AtomsToGraphs(radius=4.0,r_energy=True,r_forces=True)
#graph_data = graph_gen.convert_all(data)
#
#for conf in graph_data:
#    cell_shifts = conf.cell_offsets.float() @ conf.cell.squeeze(0)
#    conf.periodic_vec = cell_shifts
#
#sd = SimplePYG()
#torch.save(sd.collate(sd.data), "./dataset_Si_Cubuck_full_stride_20.pt")

data = ase.io.read("./gap_si_all.xyz",":")
for conf in data:
    conf.info["energy"] -= conf.get_global_number_of_atoms() * Si_REF

graph_gen = AtomsToGraphs(radius=4.0,r_energy=True,r_forces=True)
graph_data = graph_gen.convert_all(data)

for conf in graph_data:
    cell_shifts = conf.cell_offsets.float() @ conf.cell.squeeze(0)
    conf.periodic_vec = cell_shifts

sd = SimplePYG()
torch.save(sd.collate(sd.data), "./dataset_gap_Si_all_ref.pt")

# ################## load dataset ################

#class SimplePYGLoad(pyg.data.InMemoryDataset):
#    def __init__(self,transform=None, pre_transform=None, path="./data.pt"):
#        super().__init__(None, transform, pre_transform)
#        self.data, self.slices = torch.load(path)
#
#sd = SimplePYGLoad(path="./dataset_Si_Cubuck_full_stride_20.pt")

