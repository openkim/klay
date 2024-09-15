Usage
=====

Easiest way to use is to use the function :func:`klay.get_model_layers_from_yaml` to get the layers you want. You need to provide a yaml file with block titled `model`. Model block should have list of layer configs for each layer. The :func:`klay.get_model_layers_from_yaml` function will return a list of layers in sequential order. You can use the ``DETECT_PREV`` keyword in certain blocks to dynamically use the ouutput of previous layer. For example, in the following yaml file, ``linear_e3nn`` layer will use the output ``radial_basis`` layer as input.

.. code-block:: yaml

    model:
        - elem_embedding:
            embedding_type: one_hot
            n_elems: 5

        - edge_embedding:
            lmax: 6
            normalize: True
            normalization: component
            parity: True
        
        - radial_basis:
            r_max: 5.0
            num_basis: 8
            trainable: True
            power: 1
        
        - linear_e3nn:
            irreps_in: DETECT_PREV
            irreps_out: 16x0e
        
        - nequip_conv_block:
            n_conv_layers: 2
            parity: True
            lmax: 1
            conv_feature_size: 16
            node_embedding_irrep_in: DETECT_PREV
            node_attr_irrep: DETECT_PREV
            edge_attr_irrep: DETECT_PREV
            edge_embedding_irrep: DETECT_PREV
            avg_neigh: 1
            resnet: True
            radial_network_hidden_dim: 64
            radial_network_layers: 2
        
        - linear_e3nn:
            irreps_in: DETECT_PREV
            irreps_out: 1x1e

Should return 6 nequip model layers with following configuration:

.. code-block:: python

    from klay import get_model_layers_from_yaml
    model = get_model_layers_from_yaml('model.yaml')
    print(model)

.. code-block:: text

    ---------------------------------------------------------
    Generated layers with 21912 parameters
    ---------------------------------------------------------
    (0): OneHotAtomEncoding()
    (1): SphericalHarmonicEdgeAttrs(
        (sh): SphericalHarmonics()
    )
    (2): RadialBasisEdgeEncoding(
        (basis): BesselBasis()
        (cutoff): PolynomialCutoff()
    )
    (3): AtomwiseLinear(
        (linear): Linear(8x0e -> 16x0e | 128 weights)
    )
    (4): NequipConvBlock(
        (conv_layers): ModuleList(
        (0): ConvNetLayer(
            (equivariant_nonlin): Gate (32x0e+16x1o -> 16x0e+16x1o)
            (conv): InteractionBlock(
            (linear_1): Linear(16x0e -> 16x0e | 256 weights)
            (fc): FullyConnectedNet[8, 64, 64, 32]
            (tp): TensorProduct(16x0e x 1x0e+1x1o -> 16x0e+16x1o | 32 paths | 32 weights)
            (linear_2): Linear(16x0e+16x1o -> 32x0e+16x1o | 768 weights)
            (sc): FullyConnectedTensorProduct(16x0e x 1x0e -> 32x0e+16x1o | 512 paths | 512 weights)
            )
        )
        (1): ConvNetLayer(
            (equivariant_nonlin): Gate (48x0e+16x1o+16x1e -> 16x0e+16x1e+16x1o)
            (conv): InteractionBlock(
            (linear_1): Linear(16x0e+16x1o -> 16x0e+16x1o | 512 weights)
            (fc): FullyConnectedNet[8, 64, 64, 80]
            (tp): TensorProduct(16x0e+16x1o x 1x0e+1x1o -> 32x0e+32x1o+16x1e | 80 paths | 80 weights)
            (linear_2): Linear(32x0e+32x1o+16x1e -> 48x0e+16x1o+16x1e | 2304 weights)
            (sc): FullyConnectedTensorProduct(16x0e+16x1o x 1x0e -> 48x0e+16x1o+16x1e | 1024 paths | 1024 weights)
            )
        )
        )
    )
    (5): AtomwiseLinear(
        (linear): Linear(16x0e+16x1e+16x1o -> 1x1o | 16 weights)
    )
    )

You can see the example in `examples` folder for more details. You can now use these layers in your model. Below is an example of how to use these layers in a model.

.. code-block:: python

    import torch
    from klay import get_model_layers_from_yaml

    layers = get_model_layers_from_yaml('model.yaml')
    
    class Model(torch.nn.Module):
        def __init__(self, layers):
            super(Model, self).__init__()
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, z, pos, edge_index, shifts):
            node_attr = self.layers[0](z)
            edge_vec, edge_length, edge_sh = self.layers[1](pos, edge_index, shifts)
            edge_embedding = self.layers[2](edge_length)
            node_embedding = self.layers[3](node_attr)
            node_embedding = self.layers[4](node_attr, node_embedding, edge_embedding, edge_sh, edge_index)
            node_embedding = self.layers[5](node_embedding)
            return node_embedding

    model = Model(layers)

    # Example input
    z = torch.randn(10, 5)
    pos = torch.randn(10, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    shifts = torch.randn(10, 3)

    # Forward pass
    out = model(z, pos, edge_index, shifts)
    print(out)

This gives the flexibility to generate arbitrary layers and assemble them like LEGO blocks.