from klay import get_model_layers_from_yaml
import torch

layer = get_model_layers_from_yaml('model.yaml')

class Model(torch.nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, z, pos, edge_index, shift_vector):
        edge_vec, edge_length, edge_sh = self.layers[0](pos, edge_index, shift_vector)
        edge_feat = self.layers[1](edge_length)
        node_attr = self.layers[2](z)
        if node_attr.dim() == 1:
            node_attr = node_attr.unsqueeze(0)
        node_feat = self.layers[3](node_attr.to(pos.dtype))

        node_feat = self.layers[4](node_attr, node_feat, edge_feat, edge_sh, edge_index)
        forces = self.layers[5](node_feat)
        return forces
    
model = Model(layer)
model = model.to(torch.float32)
print(model)

from e3nn.util import jit

model = jit.script(model)

model.save('non_conservative_ef.pt')

# random test data
z = torch.randint(0, 5, (10,))
pos = torch.randn(10, 3)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
shift_vector = torch.randn(edge_index.size(1), 3)

print(model(z, pos, edge_index, shift_vector))