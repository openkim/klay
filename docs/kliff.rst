Training a MACE-style GNN with **klay** + **kliff**
===================================================

This tutorial shows the minimal end-to-end workflow for

1. defining a Graph Neural Network in **klay** using a YAML file,
2. compiling the model to an FX graph and TorchScripting it,
3. connecting the model to a **kliff** *Lightning* trainer,
4. running a short training loop and exporting a KIM-compatible model.

The same pattern extends to larger datasets, multi-GPU training
(`strategy="ddp"`), and more complex model graphs.

Prerequisites
-------------

* klay
* PyTorch ≥ 2.2 (built with CUDA if you train on GPU)
* e3nn -> for converting equivariant models to jit
* kliff
* torch_geometric -> for graph datasets
* torch_scatter -> torch_geometric dependency used by several layers/packages
* lightning -> for distributed GNN trainer in kliff
* tensorboard, tensorboardX -> for logging GNN trainer

You can create a valid klay + kliff env (for CPUs) using conda as:

.. code-block:: bash

    conda create -n klay-env
    conda activate klay-env
    conda install -c conda-forge python=3.9

    pip install klay
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install lightning
    pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
    pip install kliff
    pip install tensorboard tensorboardX


Directory layout used below
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   .
   ├── mace_model.yaml            # model definition (see below)
   ├── Si_training_set_4_configs  # four config files with energies/forces
   └── train_mace.py              # training script (listing follows)

.. note::

    You can use your own dataset, or download the above **toy** dataset as

.. code-block:: bash

    wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    tar -xvf Si_training_set_4_configs.tar.gz

--------------------------------------
Model definition (``mace_model.yaml``)
--------------------------------------

The YAML file enumerates **parameters**, **I/O tensors**, a **layer
graph**, and **named outputs**.  klay resolves the `${…}` references at
build time, so you only declare each hyper-parameter once.

.. code-block:: yaml

    model_params:
      r_max: 4.0
      n_channels: 32
      num_elems: 2

    model_inputs:
      species: "Tensor (N,)"
      coords: "Tensor (N,3)"
      edge_index0: "Tensor (2,E)"
      contributions: "Tensor (E,)"

    model_layers:
      element_embedding:
        type: OneHotAtomEncoding
        config: {num_elems: 2}
        inputs: {x: model_inputs.species}

      edge_feature0:
        type: SphericalHarmonicEdgeAttrs
        config: {lmax: 1}
        inputs:
          pos: model_inputs.coords
          edge_index: model_inputs.edge_index0
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
          edge_index: model_inputs.edge_index0

      output_projection:
        type: AtomwiseLinear
        config:
          irreps_in_block:
            - {"l": 0, "mul": '${model_params.n_channels}'}
            - {"l": 1, "mul": '${model_params.n_channels}'}
          irreps_out_block:
            - {"l": 0, "mul": 1}
        inputs: {h: conv1}

      contributions_energy:
        type: KIMAPISumIndex
        inputs:
          src: output_projection
          index: contributions

    model_outputs:
      energy: contributions_energy


-----------------------------------
Training script (``train_mace.py``)
-----------------------------------

The Python driver wires the model into **kliff**’s
``GNNLightningTrainer``.  All training hyper-parameters live in a single
``training_manifest`` dictionary so they are logged together and can be
re-used for checkpoint-free restarts.

.. code-block:: python

   import torch
   torch.set_default_dtype(torch.float64)

   from klay.builder import build_model
   from klay.io import load_config
   from e3nn.util import jit

   # ------------------------------------------------------------------
   # Build & script the model
   # ------------------------------------------------------------------
   mace_model = build_model(load_config("mace_model.yaml"))
   mace_model = jit.script(mace_model)          # TorchScript -> picklable, deterministic

   # ------------------------------------------------------------------
   # Experiment manifest
   # ------------------------------------------------------------------
   workspace = {"name": "GNN_train_example", "random_seed": 12345}
   dataset = {
       "type": "path",
       "path": "Si_training_set_4_configs",
       "shuffle": True
   }
   model = {"name": "MACE1",
             "input_args":
             ["species", "coords", "edge_index0", "contributions"]
   }
   transforms = {
       "configuration": {
           "name": "RadialGraph",
           "kwargs": {"cutoff": 4.0, "species": ["Si"], "n_layers": 1}
       }
   }
   training = {
       "loss": {
           "function": "MSE",
           "weights": {"config": 1.0, "energy": 1.0, "forces": 10.0},
       },
       "optimizer": {"name": "Adam", "learning_rate": 1e-3},
       "training_dataset": {"train_size": 3},
       "validation_dataset": {"val_size": 1},
       "batch_size": 1,
       "epochs": 10,
       # accelerator/strategy left on "auto" so the same script runs on CPU or GPU
       "accelerator": "auto",
       "strategy": "auto",
   }
   export = {"model_path": "./", "model_name": "MACE1__MO_111111111111_000"}

   training_manifest = {
       "workspace": workspace,
       "model": model,
       "dataset": dataset,
       "transforms": transforms,
       "training": training,
       "export": export,
   }

   # ------------------------------------------------------------------
   # Train
   # ------------------------------------------------------------------
   from kliff.trainer.lightning_trainer import GNNLightningTrainer

   trainer = GNNLightningTrainer(training_manifest, model=mace_model)
   trainer.train()
   trainer.save_kim_model()

--------------------
Running the tutorial
--------------------

.. code:: bash

   python train_mace.py           # prints a Lightning progress bar

With only four Si configurations and 10 epochs this runs in seconds on
CPU.  The call to ``save_kim_model`` writes a LAMMPS-compatible
``MACE1__MO_111111111111_000`` file plus a JSON metadata block.

Files produced
^^^^^^^^^^^^^^

* ``lightning_logs/...`` – TensorBoard logs, checkpoints
* ``MACE1__MO_111111111111_000`` – portable potential

Next steps
----------

* Swap the tiny path dataset for a real one (e.g. ANI-1x or OC20).
* Increase ``epochs`` and ``batch_size``; pick ``strategy="ddp"`` to
  distribute across multiple GPUs.
* Add more **MACE_layer** blocks or deeper radial graphs in the YAML
  to improve capacity.
* Use ``kliff``’s ``EarlyStopping`` and ``LearningRateMonitor`` callbacks
  for production runs.
