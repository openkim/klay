import torch
import numpy as np
from torch_scatter import scatter

from utils import generate_graph

# class NL_model(torch.nn.Module):
#     def __init__(self,
#             layer_dict,
#             energy_scaling_coeff=1.0, 
#             energy_shifting_coeff=0.0):
#         super().__init__()
#         self.node_attr_layer = layer_dict["elem_embed"]
#         self.node_embedding_layer = layer_dict["first_atom_embed"]
#         self.edge_attr_layer = layer_dict["edge_embed"]
#         self.edge_embedding_layer = layer_dict["radial_basis"]
#         self.conv_layers = torch.nn.ModuleList([layer_dict["conv:0"], layer_dict["conv:1"], layer_dict["conv:2"]])
#         self.post_conv_layers = torch.nn.ModuleList([layer_dict["post_conv:0"],layer_dict["post_conv:1"]])
#         self.register_buffer("energy_scaling_coeff", torch.tensor(energy_scaling_coeff))
#         self.register_buffer("energy_shifting_coeff", torch.tensor(energy_shifting_coeff))


#     def forward(self, x, pos, edge_index, period_vec, batch):
#         # elem/node embedding
#         x_attr = self.node_attr_layer(x.long().squeeze(-1))
#         x_attr = x_attr.to(dtype=pos.dtype)
#         # Linear map for embedding
#         h = self.node_embedding_layer(x_attr)

#         # Edge embedding
#         edge_vec, edge_lengths, edge_sh = self.edge_attr_layer(pos, edge_index, period_vec)

#         # Radial basis function
#         edge_length_embedding = self.edge_embedding_layer(edge_lengths)

#         # convolution
#         for layer in self.conv_layers:
#             h = layer(x_attr, h, edge_length_embedding, edge_sh, edge_index)

#         # post convolution
#         for layer in self.post_conv_layers:
#             h = layer(h)

#         h = h.squeeze()
#         # energies, forces = h[:,0], h[:,1:4]
#         # reduce energies per batch
#         h = scatter(h, batch, dim=0)
#         energies = h * self.energy_scaling_coeff + self.energy_shifting_coeff
#         energies = energies.unsqueeze(-1)
#         return energies

class RepModule(torch.nn.Module):
    def __init__(self, layer_dict):
        super().__init__()
        self.node_attr_layer = layer_dict["elem_embed"]
        self.node_embedding_layer = layer_dict["first_atom_embed"]
        self.edge_attr_layer = layer_dict["edge_embed"]
        self.edge_embedding_layer = layer_dict["radial_basis"]
        self.conv_layers = torch.nn.ModuleList([layer_dict["conv:0"], layer_dict["conv:1"], layer_dict["conv:2"]])

    def forward(self, x, pos, edge_index, period_vec, batch):
        # elem/node embedding
        x_attr = self.node_attr_layer(x.long().squeeze(-1))
        x_attr = x_attr.to(dtype=pos.dtype)
        # Linear map for embedding
        h = self.node_embedding_layer(x_attr)

        # Edge embedding
        edge_vec, edge_lengths, edge_sh = self.edge_attr_layer(pos, edge_index, period_vec)

        # Radial basis function
        edge_length_embedding = self.edge_embedding_layer(edge_lengths)

        # convolution
        for layer in self.conv_layers:
            h = layer(x_attr, h, edge_length_embedding, edge_sh, edge_index)
        
        # return atomic representations
        return h # (num_atoms, dims)

class OutputModule(torch.nn.Module):
    def __init__(self,
        model_config,
        layer_dict,
        energy_scaling_coeff: torch.Tensor,
        energy_shifting_coeff: torch.Tensor
    ):
        super().__init__()
        self.model_config = model_config
        self.num_targets = model_config["num_targets"]

        # self.post_conv_activtaion = layer_dict["post_conv_activation"]
        self.post_conv_layers = torch.nn.ModuleList([layer_dict["post_conv:0"], layer_dict["post_conv:1"]])

        self.register_buffer("energy_scaling_coeff", energy_scaling_coeff)
        self.register_buffer("energy_shifting_coeff", energy_shifting_coeff)

    def forward(self, h, batch):
        # first linear layer that maps to different heads
        # h = self.post_conv_layers[0](h)
        # h = h.view(-1, self.num_targets, self.model_config["post_conv_irreps"][0]) # (num_atoms, num_targets, dims)

        # post convolution
        for layer in self.post_conv_layers:
            h = layer(h)

        h = h.squeeze(-1) # (num_atoms, num_targets)
        h = scatter(h, batch, dim=0, reduce='sum') # (num_batches, num_targets)

        energies = h * self.energy_scaling_coeff + self.energy_shifting_coeff
        return energies
    
class NL_model(torch.nn.Module):
    def __init__(self,
        model_config,
        layer_dict,
        energy_scaling_coeff: dict, 
        energy_shifting_coeff: dict
    ):
        super().__init__()
        self.model_config = model_config
        
        assert self.model_config["num_targets"] == len(energy_scaling_coeff), \
            "Number of energy scaling coefficients must match number of heads."

        # map dataset names to dimensions
        self.idx2dataset = {i: dataset for i, dataset in enumerate(self.model_config["datasets"])}
        self.dataset2idx = {dataset: i for i, dataset in enumerate(self.model_config["datasets"])}

        energy_scaling_coeff_tensor = torch.tensor([[
            energy_scaling_coeff.get(dataset, 1.0) for dataset in self.model_config["datasets"]
        ]])
        energy_shifting_coeff_tensor = torch.tensor([[
            energy_shifting_coeff.get(dataset, 0.0) for dataset in self.model_config["datasets"]
        ]])

        self.rep_module = RepModule(layer_dict)
        self.out_module = OutputModule(
            model_config, layer_dict, energy_scaling_coeff_tensor, energy_shifting_coeff_tensor
        )

    # def forward(self, x, pos, edge_index, period_vec, batch, per_config_dataset_idx):
    #     if edge_index is None:
    #         edge_index = generate_graph(
    #             data,
    #             cutoff=None,
    #             max_neighbors=None,
    #             use_pbc=None,
    #             otf_graph=None,
    #             enforce_max_neighbors_strictly=None
    #         )
            
    #     h = self.rep_module(x, pos, edge_index, period_vec, batch)
    #     energies = self.out_module(h, batch)

    #     batch_index = torch.arange(energies.shape[0], device=energies.device)
    #     energies = energies[batch_index, per_config_dataset_idx]
    #     forces = -torch.autograd.grad(
    #         energies, pos, create_graph=True, 
    #         grad_outputs=torch.ones_like(energies), allow_unused=True)[0] # (num_atoms, 3)
        
    #     # forces = []
    #     # # compute per head forces
    #     # for i in range(energies.shape[1]):
    #     #     E = energies[:, i] # (num_batches)
    #     #     F = -torch.autograd.grad(
    #     #         E, pos, create_graph=True, 
    #     #         grad_outputs=torch.ones_like(E), allow_unused=True)[0] # (num_atoms, 3)
    #     #     forces.append(F)
    #     # forces = torch.stack(forces, dim=1) # (num_atoms, num_targets, 3)

    #     return energies, forces
    
    def forward(self, data, per_config_dataset_idx):
        x, pos, batch = data.tags, data.pos.requires_grad_(True), data.batch

        if not hasattr(data, "edge_index") or data.edge_index is None:
            (
                edge_index,
                edge_dist,
                distance_vec,
                cell_offsets,
                period_vec,
                neighbors
            ) = generate_graph(
                data,
                cutoff=6,
                max_neighbors=1000,
                use_pbc=True,
                otf_graph=True,
                enforce_max_neighbors_strictly=False
            )
        else:
            edge_index, period_vec = data.edge_index, data.periodic_vec

        # sort edge_index by the second column then the first column
        # edge_index_np = edge_index.cpu().numpy().T
        # edge_index_np = edge_index_np[np.lexsort((edge_index_np[:,1], edge_index_np[:,0]))].T
        # edge_index = torch.from_numpy(edge_index_np).to(pos.device)

        # if not torch.equal(edge_index[0], edge_index_new[1]) and not torch.equal(edge_index[1], edge_index_new[0]):
        #     print(edge_index.shape, edge_index_new.shape) 
        #     print(edge_index)
        #     print(edge_index_new)
        #     print("Edge index mismatch.")
        #     raise ValueError("Edge index mismatch.")
        # # assert torch.equal(edge_index, edge_index_new), "Edge index mismatch."
        
        h = self.rep_module(x, pos, edge_index, period_vec, batch)
        energies = self.out_module(h, batch)

        batch_index = torch.arange(energies.shape[0], device=energies.device)
        energies = energies[batch_index, per_config_dataset_idx]
        forces = -torch.autograd.grad(
            energies, pos, create_graph=True, 
            grad_outputs=torch.ones_like(energies), allow_unused=True)[0] # (num_atoms, 3)
        
        # forces = []
        # # compute per head forces
        # for i in range(energies.shape[1]):
        #     E = energies[:, i] # (num_batches)
        #     F = -torch.autograd.grad(
        #         E, pos, create_graph=True, 
        #         grad_outputs=torch.ones_like(E), allow_unused=True)[0] # (num_atoms, 3)
        #     forces.append(F)
        # forces = torch.stack(forces, dim=1) # (num_atoms, num_targets, 3)

        return energies, forces