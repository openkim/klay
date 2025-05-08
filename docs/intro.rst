.. _intro:

KLay – composable layers for MLIPs
===============================

``KLay`` is a lightweight toolbox that lets you **design, visualise and export
neural-network layers** for machine-learning interatomic potentials (MLIPs)
*without* writing boiler-plate model code. It was started as a module for KLIFF,
but given the complexity, we decided to extract it into a separate package.
It is designed to ensure that all generated ML models are compatible with the
KLIFF-OpenKIM-TorchML framework.

Why another library?
--------------------

Most MLIP packages ship monolithic models: great for a quick benchmark,
tedious to modify.  KLay takes the complementary “LEGO-block” approach:

* **Every layer is first-class.**  You can register a new module with a single
  decorator, override its constructor via ``from_config`` and use it
  immediately in YAML.
* **Graphs are transparent.**  A YAML file is turned into a NetworkX
  **DAG**, then into a `torch.fx.GraphModule`.  You can draw the graph,
  rewrite it, quantise it or feed it to `torch.compile` [Experimental].
* **Nothing is hidden.**  All building blocks are plain `torch.nn.Module`
  subclasses.  Training with **KLIFF**, standalone PyTorch, or even pure
  TorchScript is possible out-of-the-box.

Key features
============

* OmegaConf/YAML schema with variable interpolation.
* **Alias call-sites** – call the *same* module multiple times → shared
  weights.
* Friendly **CLI**: ``klay validate``, ``klay layers``, ``klay export``.
* Immediate **autograd force layer** *(optional second-order graph)*.
* Extensible registry with coloured table output for documentation.

Quick installation
------------------

.. code-block:: bash

   pip install klay              # latest release
   # or: bleeding-edge
   pip install git+https://github.com/openkim/klay.git

One-minute example
==================

.. code-block:: python

   import torch
   from klay.io import load_config
   from klay.builder  import build_model

   cfg   = load_config("examples/mace.yaml")
   model = build_model(cfg)           # GraphModule
   N, E  = 4, 8
   batch = dict(
       atomic_numbers = torch.randint(1, 3, (N,)),
       positions      = torch.rand(N, 3, requires_grad=True),
       edge_index     = torch.randint(0, N, (2, E)),
       shifts         = torch.zeros(E, 3),
   )
   out = model(**batch)
   print(out["energy"], out["forces"].shape)     # scalar,  (N,3)


Architecture in a nutshell
==========================

.. mermaid::

   graph LR
       subgraph "OmegaConf / YAML"
          X(model_inputs) --> Y(model_layers) --> Z(model_outputs)
       end
       Z --> D(DAG / NetworkX)
       D --> FX(torch.fx.GraphModule)
       FX -->|TorchScript| PT(.pt)
       FX -->|torch.compile| JIT(Optimised model)

* **YAML → DAG** edges = data-dependencies, nodes are inputs / layers /
  call-site aliases / outputs.
* **DAG → FX** one placeholder per ``model_input``, one ``call_module`` node
  per layer, auto-generated ``getitem`` nodes for tuple/dict ports.
* **FX → Deploy** TorchScript, `torch.compile`, ONNX, etc.

Command-line highlights
=======================

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
     - Frequent options
   * - ``klay layers``
     - Show every registered layer and its ``from_config`` signature.  Required
       args are **red**, optional args (with defaults) **yellow**.
     - ``--type embedding`` ``--all``
   * - ``klay validate``
     - Fatal on missing sources, cycles; warnings for alias→alias, unused
       outputs, dangling layers (optionally fatal).  Can render a Graphviz
       image.
     - ``--allow-dangling`` ``--visualize`` ``--fmt svg``
   * - ``klay build-layers``
     - Instantiate all modules when the config lacks inputs/outputs.
     - —
   * - ``klay export``
     - Build the model and save either a TorchScript ``.pt`` file or a weights
       ``.pth``.
     - ``--format state_dict`` ``-n 10``
