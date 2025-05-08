.. _usage:

==========================
Using KLay
==========================

This page explains **every section of a KLay YAML file**, how the fields fit
together, and the shorthand notations you can use to keep configs concise.

A complete, minimal example is shown first; each part is dissected in the
sections that follow.

.. code-block:: yaml

    model_params:
      r_max: 4.0
      n_channels: 32
      num_elems: 2

    model_inputs:
      atomic_numbers: "Tensor (N,)"
      positions: "Tensor (N,3)"
      edge_index: "Tensor (2,E)"

    model_layers:
      element_embedding:
        type: OneHotAtomEncoding
        config: {num_elems: 2}
        inputs: {x: model_inputs.atomic_numbers}

      edge_feature0:
        type: SphericalHarmonicEdgeAttrs
        config: {lmax: 1}
        inputs:
          pos: model_inputs.positions
          edge_index: model_inputs.edge_index
        output: {0: vec0, 1: len0, 2: sh0}

      radial_basis_func:
        type: RadialBasisEdgeEncoding
        config:
          r_max: ${model_params.r_max}
        inputs:
          edge_length: len0

      node_features:
        type: AtomwiseLinear
        config:
          irreps_in_block:
            - {"l": 0, "mul": '${model_params.num_elems}'}
          irreps_out_block:
            - {"l": 0, "mul": '${model_params.n_channels}'}
        inputs: {h: element_embedding}

      conv1:
        type: MACE_layer
        config:
          lmax: 1
          correlation: 2
          num_elements: ${model_params.num_elems}
          hidden_irreps_block:
            - {"l": 0, "mul": '${model_params.n_channels}'}
            - {"l": 1, "mul": '${model_params.n_channels}'}
          input_block: ${model_layers.node_features.config.irreps_out_block}
          node_attr_block: ${model_layers.node_features.config.irreps_in_block}
        inputs:
          vectors: vec0
          node_feats: node_features
          node_attrs: element_embedding
          edge_feats: radial_basis_func
          edge_index: model_inputs.edge_index

      output_projection:
        type: AtomwiseLinear
        config:
          irreps_in_block:
            - {"l": 0, "mul": '${model_params.n_channels}'}
            - {"l": 1, "mul": '${model_params.n_channels}'}
          irreps_out_block:
            - {"l": 0, "mul": 1}
        inputs: {h: conv1}

      forces:
        type: AutogradForces
        inputs:
          energy: output_projection
          pos: model_inputs.positions

    model_outputs:
      energy: output_projection
      forces: forces
      representation: conv1



----------------------------------------------------------------
1.  ``model_params`` – hyper-parameters & shared constants
----------------------------------------------------------------

``model_params`` is a free-form mapping of names → values.
Any value can be *referenced* later with OmegaConf interpolation::

   ${model_params.r_max}

Typical items:

* cut-off radii (``r_max``), channel counts, element counts
* Booleans (``use_pbc``), learning-rate schedules …

----------------------------------------------------------------
2.  ``model_inputs`` – declare the forward() signature
----------------------------------------------------------------

Keys become **argument names** of the generated model:

.. code-block:: yaml

   model_inputs:
     atomic_numbers:  "Tensor (N,)"
     positions:       "Tensor (N,3)"
     edge_index:      "Tensor (2,E)"
     shifts:          "Tensor (E,3)"     # optional for PBC

*The value is just a comment; KLay never parses it.*

----------------------------------------------------------------
3.  ``model_layers`` – the heart of the graph
----------------------------------------------------------------

Each top-level key creates **one node** in the DAG.

Layer declaration
=================

.. code-block:: yaml

   element_embedding:
     type: OneHotAtomEncoding           # registry key
     config: {num_elems: ${model_params.num_elems}}
     inputs: {x: model_inputs.atomic_numbers}

Keys
~~~~

+ ``type``   | registry key (see ``klay layers``)
+ ``config`` | kwargs passed to ``from_config`` or ``__init__``
+ ``inputs`` | mapping *port -> source*
+ ``output`` | *optional* mapping *inner-key (or index) -> alias*
+ ``alias``  | *optional* — treat this entry as **another call-site** for an
              already-declared module (shared weights)

Input references
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Syntax
     - Meaning
   * - ``model_inputs.x``
     - forward() argument ``x``
   * - ``layer_name``
     - whole output of that layer
   * - ``layer_name.k``
     - *k-th* tuple element **or** dict key ``k``
   * - any alias
     - whatever ``output:`` mapped under that name

Irreducible–representation (irrep) blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

klay follows the same compact string format as **e3nn** but lets you
declare each block as a dictionary—convenient for YAML and
programmatic construction. This ensures readability and abstracts away any e3nn specific
notation.

Each block has

* ``l`` – the total angular-momentum order (integer ≥ 0)
* ``mul`` – multiplicity, i.e. how many copies of that irrep
* optional ``p`` – explicit parity, ``"even"`` or ``"odd"``

Default parity
--------------

If the ``p`` key is omitted, klay assigns it automatically:

* ``e`` (even) when ``l`` is even
* ``o`` (odd)  when ``l`` is odd

Example
-------

.. code-block:: python

   blocks = [
       {"l": 0, "mul": 64},                # 0  -> default even  -> 64x0e
       {"l": 1, "mul": 32},                # 1  -> default odd   -> 32x1o
       {"l": 2, "mul": 16, "p": "odd"},    # explicit override -> 16x2o
   ]

   >>> irreps_blocks_to_string(blocks)
   '64x0e + 32x1o + 16x2o'

Copy-and-paste the block list into your ``*.yaml`` model file or build
it on the fly in Python; klay will convert it to the canonical irrep
string wherever an ``irreps_*`` field is expected.


Output mapping
~~~~~~~~~~~~~~

Use it when a layer returns a **tuple** or **dict** and you want human-readable
names:

.. code-block:: yaml

   edge_feature0:
     type: SphericalHarmonicEdgeAttrs
     output:
       0: vec0          # tuple index -> alias
       1: len0
       2: sh0

KLay auto-creates *both* the alias (``len0``) **and** ``edge_feature0.1``
reference.

Alias (weight sharing, Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   edge_conv:                       # declares module
     type: ConvNetLayer
     config: {…}

   conv1:                           # extra call (shares weights)
     alias: edge_conv
     inputs: {h: node_features, edge_sh: sh0}

----------------------------------------------------------------
4.  ``model_outputs`` – what you want out
----------------------------------------------------------------

A **mapping** where the key is the public name and the value is a reference:

.. code-block:: yaml

   model_outputs:
     energy:          output_projection
     forces:          forces
     representation:  conv1

* A trailing ``.k`` or ``.key`` selects a field from tuple/dict.
* If you omit ``model_inputs`` *and* ``model_outputs`` the builder returns a
  dict ``{name: nn.Module}`` – handy for “layer library” configs.

----------------------------------------------------------------
Cheat-sheet
----------------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Syntax
     - Expands to
     - Use-case
   * - ``model_inputs.pos``
     - placeholder tensor ``pos``
     - raw model input
   * - ``layerA``
     - full output of ``layerA``
     - single-output layers
   * - ``layerA.2``
     - 3ʳᵈ tuple slot of ``layerA``
     - tuple-return layers
   * - ``layerA.edge_len``
     - dict key ``"edge_len"`` of ``layerA``
     - dict-return layers
   * - ``alias_name``
     - whatever ``output:`` mapped to that alias
     - readable wiring

----------------------------------------------------------------
Gotchas & best practices
----------------------------------------------------------------

* **Forces training** – set ``create_graph: true`` in the
  ``ForceFromEnergy`` layer to keep second-order derivatives.
* **Avoid dotted aliases** for clarity: prefer ``edge_length: len0`` over
  ``edge_feature0.1``.
* **Validate & draw** your graph before training:

  .. code-block:: bash

     klay validate model.yml -v

* **Export** a stand-alone TorchScript model:

  .. code-block:: bash

     klay export model.yml -o model.pt
