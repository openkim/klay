import ase
import ase.neighborlist
import torch
import numpy as np

from simple_nequip import gen_model

torch.set_default_dtype(torch.float64)

np.random.seed(0)

def generate_random_2d_lattices():
    """
    Generates a random 2D lattice of atoms
    """
    n_particles = np.random.randint(10, 25)
    pos = np.random.rand(n_particles, 3)
    pos[:, 0:2] *= 10
    pos[:, 2] = 0
    cell = np.array([[10., 0, 0], [0, 10., 0], [0, 0, 0]]) # cell size 10 10 0
    return  pos, cell


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    # ASE dependent part
    cell = ase.geometry.complete_cell(cell)
    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc,
            cell,
            pos,
            cutoff=r_max,
            self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
            use_scaled_positions=False,
        )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = np.vstack((first_idex, second_idex))

    return edge_index, shifts, cell

model = gen_model("in.yaml", save=False)

dataset_size = 10
epochs = 10
r_max = 3.0

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(dataset_size):
    optim.zero_grad()
    pos, cell = generate_random_2d_lattices()
    edge_index, shifts, cell = neighbor_list_and_relative_vec(pos, r_max, cell=cell, pbc=[True, True, False]) 
    pos = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    shifts = torch.tensor(shifts @ cell) # shifts in periodic vectors
    
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    species = torch.zeros(pos.shape[0], dtype=torch.long)
    
    out = model(species, pos, edge_index, shifts, batch) #shape: (n_atoms, 3)

    dummy_ground_truth = torch.rand(out.shape[0], 3)
    loss = torch.nn.MSELoss()(out, dummy_ground_truth)
    loss.backward()
    optim.step()
    print("Finished iteration", i, loss.item())