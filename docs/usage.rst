.. _usage:

==========================
Using KLay
==========================

This page explains **every section of a KLay YAML file**, how the fields fit
together, and the shorthand notations you can use to keep configs concise.

A complete, minimal example is shown first; each part is dissected in the
sections that follow.

.. literalinclude:: ../example/mace_model.yaml
   :language: yaml
   :caption: *A working MACE-style energy + force model.*

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

+ ``type``  | registry key (see ``klay layers``)
+ ``config``| kwargs passed to ``from_config`` or ``__init__``
+ ``inputs``| mapping *port -> source*
+ ``output``| *optional* mapping *inner-key (or index) -> alias*
+ ``alias`` | *optional* — treat this entry as **another call-site** for an
              already-declared module (shared weights)

Input references
~~~~~~~~~~~~~~~~

===============  =========================================
Form             Meaning
===============  =========================================
``model_inputs.x`` | forward() argument ``x``
``layer_name``     | whole output of that layer
``layer_name.k``   | *k-th* tuple element **or** dict key ``k``
any alias          | whatever ``output:`` mapped under that name
===============  =========================================

Output mapping
~~~~~~~~~~~~~~

Use it when a layer returns a **tuple** or **dict** and you want human-readable
names:

.. code-block:: yaml

   edge_feature0:
     type: SphericalHarmonicEdgeAttrs
     output:
       0: vec0          # tuple index → alias
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
