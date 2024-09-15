API Reference
=============

This document provides a reference for the main functions in the module.

Enums
-----

Layers
~~~~~~

.. py:class:: Layers

   An enumeration of supported layer types.

   .. py:attribute:: ELEM_EMBEDDING = 0
   .. py:attribute:: EDGE_EMBEDDING = 1
   .. py:attribute:: RADIAL_BASIS = 2
   .. py:attribute:: LINEAR_E3NN = 3
   .. py:attribute:: NEQUIP_CONV = 4
   .. py:attribute:: NEQUIP_CONV_BLOCK = 5

ElemEmbedding
~~~~~~~~~~~~~

.. py:class:: ElemEmbedding

   An enumeration of supported element embedding types.

   .. py:attribute:: ONE_HOT = 0
   .. py:attribute:: BINARY = 1
   .. py:attribute:: ELECTRON = 2

   .. py:method:: get_embed_type_from_str(embed_str: str) -> ElemEmbedding

      Get element embedding type from string.

      :param embed_str: Element embedding type as a string.
      :return: Corresponding ElemEmbedding enum value.

Functions
---------

.. py:function:: summary()

   Print a summary of supported layers and their required parameters.

.. py:function:: get_element_embedding(embedding_type: str, n_elems: int = 118) -> torch.nn.Module

   Get torch module for element embedding.

   :param embedding_type: Element embedding type.
   :param n_elems: Number of elements (only for one_hot).
   :return: Element embedding module.

.. py:function:: get_edge_embedding(lmax: int, normalize: bool = True, normalization: str = "component", parity: bool = True) -> torch.nn.Module

   Returns edge embedding module.

   :param lmax: Maximum l value for spherical harmonics.
   :param normalize: Whether to normalize the spherical harmonics.
   :param normalization: Normalization scheme to use.
   :param parity: Whether to use parity.
   :return: Edge embedding module.

.. py:function:: get_radial_basis(r_max: Union[float, torch.Tensor], num_basis: int = 8, trainable: bool = True, power: int = 6) -> torch.nn.Module

   Returns radial basis module.

   :param r_max: Cutoff radius.
   :param num_basis: Number of basis functions.
   :param trainable: Whether the basis functions are trainable.
   :param power: Power used in envelope function.
   :return: Radial basis module.

.. py:function:: get_linear_e3nn(irreps_in, irreps_out) -> torch.nn.Module

   Get linear e3nn module.

   :param irreps_in: Input irreps.
   :param irreps_out: Output irreps.
   :return: Linear e3nn module.

.. py:function:: get_nequip_conv(parity: bool, lmax: int, conv_feature_size: int, node_embedding_irrep_in, node_attr_irrep, edge_attr_irrep, edge_embedding_irrep, avg_neigh=1, nonlinearity_type="gate", resnet=False, nonlinearity_scalars={"e": "ssp", "o": "tanh"}, nonlinearity_gates={"e": "ssp", "o": "abs"}, radial_network_hidden_dim=64, radial_network_layers=2) -> torch.nn.Module

   Get NequIP convolution layer.

   :param parity: Whether to use parity.
   :param lmax: Maximum l value for spherical harmonics.
   :param conv_feature_size: Convolution feature size.
   :param node_embedding_irrep_in: Input node embedding irreps.
   :param node_attr_irrep: Node attribute irreps.
   :param edge_attr_irrep: Edge attribute irreps.
   :param edge_embedding_irrep: Edge embedding irreps.
   :param avg_neigh: Average number of neighbors.
   :param nonlinearity_type: Nonlinearity type.
   :param resnet: Whether to use resnet.
   :param nonlinearity_scalars: Nonlinearity scalars.
   :param nonlinearity_gates: Nonlinearity gates.
   :param radial_network_hidden_dim: Radial network hidden dimension.
   :param radial_network_layers: Radial network layers.
   :return: NequIP convolution layer.

.. py:function:: get_nequip_conv_block(n_conv_layers: int, parity: bool, lmax: int, conv_feature_size: int, node_embedding_irrep_in, node_attr_irrep, edge_attr_irrep, edge_embedding_irrep, avg_neigh=1, nonlinearity_type="gate", resnet=False, nonlinearity_scalars={"e": "ssp", "o": "tanh"}, nonlinearity_gates={"e": "ssp", "o": "abs"}, radial_network_hidden_dim=64, radial_network_layers=2) -> torch.nn.Module

   Returns NequIP convolution block, with multiple convolution layers.

   :param n_conv_layers: Number of conv layers.
   :param parity: Whether to use parity.
   :param lmax: Maximum l value for spherical harmonics.
   :param conv_feature_size: Convolution feature size.
   :param node_embedding_irrep_in: Input node embedding irreps.
   :param node_attr_irrep: Node attribute irreps.
   :param edge_attr_irrep: Edge attribute irreps.
   :param edge_embedding_irrep: Edge embedding irreps.
   :param avg_neigh: Average number of neighbors.
   :param nonlinearity_type: Nonlinearity type.
   :param resnet: Whether to use resnet.
   :param nonlinearity_scalars: Nonlinearity scalars.
   :param nonlinearity_gates: Nonlinearity gates.
   :param radial_network_hidden_dim: Radial network hidden dimension.
   :param radial_network_layers: Radial network layers.
   :return: NequIP convolution block.

.. py:function:: get_model_from_yaml(yaml_file) -> torch.nn.Sequential

   Generate model sequentially from yaml file.

   :param yaml_file: Path to yaml file.
   :return: Sequential model.

   