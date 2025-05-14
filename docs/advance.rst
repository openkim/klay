=================
Advanced Models
=================

Klay’s registry already covers the most common GNN, embedding‐ and
convolutional blocks used in MLIPs and is actively curated for more layers,
but advanced research often calls for **arbitrary PyTorch callables** and
**weight-sharing across multiple stages**.
This page shows how to achieve both with two tiny YAML constructs:

* ``ArbitraryModule`` – wrap *any* import-path (class **or** function).
* **Aliases** – call an **existing** layer again, enabling parallel or
  staged graphs with shared parameters.

---------------------------------------------------
End-to-end example
---------------------------------------------------

.. code-block:: yaml

   model_inputs:
     x: "(N,16) tensor you feed at runtime"

   model_layers:
     dense:                    # one Linear layer
       type: ArbitraryModule
       config:
         target: torch.nn.Linear
         args:  [16, 16]
       inputs: {0: model_inputs.x}

     relu:                     # ReLU module
       type: ArbitraryModule
       config: {target: torch.nn.ReLU}
       inputs: {0: dense}

     second_pass:              #  *alias* -> reuse the same Linear
       alias: dense
       inputs: {0: relu}

     relu2:                    #  functional ReLU
       type: ArbitraryModule
       config: {target: torch.nn.functional.relu}
       inputs: {0: second_pass}

   model_outputs:
     preds: relu2

Run it:

.. code-block:: python

   from klay.builder import build_model
   from klay.io      import load_config
   import torch

   cfg   = load_config("example/arbitrary_and_alias.yaml")
   model = build_model(cfg)
   out = model(torch.rand(16))      # torch.Size([16])

----------------------------------------------------------------
1  Arbitrary Layers  (`ArbitraryModule`)
----------------------------------------------------------------

``ArbitraryModule`` turns **any importable Python callable** into a Klay
layer.  Its ``config`` block mirrors a constructor call:

.. code-block:: yaml

   some_layer:
     type:  ArbitraryModule
     config:
       target: torch.nn.functional.gelu   # dotted import path
       args: []                           # -> *positional* args
       kwargs: {}                         # -> **keyword** args
     inputs:
       0: previous_tensor                 # maps to arg 0
     # output map is optional; omit for single-tensor returns

How it works
~~~~~~~~~~~~

* If ``target`` resolves to a **class** derived from
  ``torch.nn.Module``. It is **instantiated** with the given
  ``args/kwargs``.
* Otherwise the callable is kept **intact**; when ``args`` or ``kwargs``
  are non-empty Klay wraps it in :pymod:`functools.partial`.
* The wrapper still behaves like a standard layer, so you can **alias**
  it, trace it with :pyclass:`torch.fx.GraphModule`, and place it in
  any branch of a larger DAG.

Input block rules
~~~~~~~~~~~~~~~~~

* **Positional ports** use integer keys (``0``, ``1``...).
  They are forwarded in sorted order.
  Keep them consistent with the callable’s signature.
* **Keyword ports** use strings; they map 1-to-1 to argument names.

------------------------------------------------------------------
2  Layer Aliases  (weight sharing & staged graphs)
------------------------------------------------------------------

An *alias* creates a **second call-site** to an **already-declared**
layer, reusing its parameters:

.. code-block:: yaml

   next_stage:
     alias: previous_dense     # must point to an existing layer name
     inputs:
       0: some_tensor          # new data path
     output: {0: stage_out}

Typical use-cases
~~~~~~~~~~~~~~~~~

* **Parallel branches** – e.g. self-attention where the same MLP block
  serves query/key/value heads.
* **Staged graphs** – run the same block (e.g. embedding) on different input graphs
  for a domain decomposition invariant parallel staged graphs to be used in OpenKIM
  `TorchML driver <https://openkim.org/id/TorchML__MD_173118614730_001>`_.
* **Recurrent constructs** – feed the output of a block back into
  itself for another iteration.

Key points
~~~~~~~~~~

* The alias appears as a normal node in the FX graph, but **shares** the
  original parameters (``state_dict`` stores them only once).
* You may define a fresh ``inputs`` / ``output`` map to wire the alias
  into a new context.
* **Validation** – the DAG builder ensures ``alias`` targets an existing
  layer and that no cycles are introduced.

---------------------------------------------------
Caveats & compliance notes
---------------------------------------------------

.. warning::

   * **TorchScript support is *best-effort*.**
     Arbitrary callables that rely on Python features not supported by
     TorchScript (e.g. string operations, dynamic shapes) will break
     serialization.

   * **OpenKIM validators require scripted models.**
     If you intend to register a potential under the KIM framework,
     re-implement your arbitrary or functional layers as
     ``torch.nn.Module`` and confirm they script cleanly.

   * **Functional callables + constructor args**
     Remember to provide ``args`` / ``kwargs`` *only* when the function
     actually expects them; mismatched signatures surface at run-time.

   * **Alias loops** are disallowed.  The DAG check stops you from
     creating recursive references, but be mindful when chaining many
     aliases in complex graphs.


.. tip::
    **This should be enough for most complicated MLIPs, but if it is not, you can always use
    KLay to generate layers and manually build the model yourself.**
