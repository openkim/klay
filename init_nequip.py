import yaml
import sys
import layers as l
import torch
torch.set_default_dtype(torch.float64)
from e3nn import o3
from e3nn.util import jit
from e3nn import nn as enn

yaml_file = sys.argv[1]
# file data to dict
with open(yaml_file, 'r') as stream:
    data = yaml.safe_load(stream)
print(f"read data: {data}")
# Radial Layers
radial_encoding = l.RadialBasisEdgeEncoding(basis_kwargs={"r_max": data["cutoff"], "num_basis":data['n_radial_basis']},
        cutoff_kwargs={"r_max": data["cutoff"]})
# test
x = torch.rand(1,3)
print(f"Radial test: {radial_encoding(x)}")

# Edge encoding (bonds)
edge_encoding = l.SphericalHarmonicEdgeAttrs(data["edge_irreps"])
# test
pos = torch.rand(2,3)
edge_index = torch.tensor([[0,1],[1,0]])
print(f"Edge test: {edge_encoding(pos, edge_index)}")

# element embedding
n_elem = len(data["elements"])
elem_emb = l.OneHotAtomEncoding(n_elem)
# test
elem = torch.tensor([0])
print(f"Element test: {elem_emb(elem)}")
node_out_attr = f"{n_elem}x0e"

# node attributes
node_attr = o3.Linear(node_out_attr, data["node_irreps"])
# test
x = torch.tensor([[1.0]])
print(f"Node test: {node_attr(x)}")

# Convolution Layers
node_attr_irrep = f"{n_elem}x0e"
node_embedding_irrep = data["node_irreps"]
edge_attr_irrep = data["edge_irreps"]
radial_attr_irrep = f"{data['n_radial_basis']}x0e"
raidal_nn_layers = data["invariant_layers"]
raidal_nn_neurons = data["invariant_neurons"]
node_embedding_out_irrep = data["hidden_conv_irreps"][0]
conv = l.ConvNetLayer(node_attr_irrep, node_embedding_irrep, edge_attr_irrep, radial_attr_irrep, raidal_nn_layers, raidal_nn_neurons, node_embedding_out_irrep)

print(conv)
# Final linear layer for transform to scalar
final_linear = o3.Linear(data["hidden_conv_irreps"][-1], data["out_conv_irreps"])
final_non_linearity = enn.NormActivation(data["out_conv_irreps"], torch.tanh)
# test
print("final linear", final_linear)
print(final_non_linearity)

# 3 Linear layers for varying resnet like parameters
rnl1 = o3.Linear(data["out_conv_irreps"], data["out_conv_irreps"])
rnl2 = o3.Linear(data["out_conv_irreps"], data["out_conv_irreps"])
rnl3 = o3.Linear(data["out_conv_irreps"], data["out_conv_irreps"])


rnl1_nl = enn.NormActivation(data["out_conv_irreps"], torch.tanh)
rnl2_nl = enn.NormActivation(data["out_conv_irreps"], torch.tanh)
rnl3_nl = enn.NormActivation(data["out_conv_irreps"], torch.tanh)
print(rnl1)

# Final nn mapping to scalars
nn_map = enn.FullyConnectedNet([32,64,32,1], act=torch.tanh)
print(nn_map)

# Model E3NN class
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radial_encoding = radial_encoding
        self.edge_encoding = edge_encoding
        self.elem_emb = elem_emb
        self.node_attr = node_attr
        self.conv = conv
        self.final_linear = final_linear
        self.final_non_linearity = final_non_linearity
        self.rnl1 = rnl1
        self.rnl2 = rnl2
        self.rnl3 = rnl3
        self.rnl1_nl = rnl1_nl
        self.rnl2_nl = rnl2_nl
        self.rnl3_nl = rnl3_nl
        self.nn_map = nn_map

    def forward(self, species, coords, edge_index, contributions):
        # Edge encoding
        edge_vec, edge_embed = self.edge_encoding(coords, edge_index)
        
        # Radial encoding
        edge_length, edge_attr = self.radial_encoding(edge_vec)
        
        # Element embedding
        elem = self.elem_emb(species)
        
        # Node attributes
        node_attr = self.node_attr(elem)
        
        # Convolution
        x:torch.Tensor = self.conv(elem, node_attr, edge_embed, edge_attr, edge_index)
        
        # Final linear layer
        x = self.final_linear(x)
        x = self.final_non_linearity(x)
        
        # Resnet like
        x = x + self.rnl1_nl(self.rnl1(x))
        #x = x + self.rnl2_nl(self.rnl2(x))
        #x = x + self.rnl3_nl(self.rnl3(x))

        # Final nn mapping
        x = self.nn_map(x)
        return x

model = Model()
print(model)
model = jit.script(model)
model.save("model.pt")
