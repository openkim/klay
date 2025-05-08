import torch

torch.set_default_dtype(torch.float64)

from klay.builder import build_model
from klay.io import load_config

mace_model = build_model(load_config("./mace_model.yaml"))

from e3nn.util import jit

mace_model = jit.script(mace_model)


workspace = {"name": "GNN_train_example", "random_seed": 12345}
dataset = {"type": "path", "path": "Si_training_set_4_configs", "shuffle": True}
model = {"name": "MACE1", "input_args": ["species", "coords", "edge_index0"]}
transforms = {
    "configuration": {
        "name": "RadialGraph",
        "kwargs": {"cutoff": 4.0, "species": ["Si"], "n_layers": 1},
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
}
training["strategy"] = "auto"  # only for jupyter notebook, try "auto" or "ddp" for normal usage
training["accelerator"] = "auto"  # for Apple Mac, "auto" for rest

export = {"model_path": "./", "model_name": "MACE1__MO_111111111111_000"}

training_manifest = {
    "workspace": workspace,
    "model": model,
    "dataset": dataset,
    "transforms": transforms,
    "training": training,
    "export": export,
}

from kliff.trainer.lightning_trainer import GNNLightningTrainer

trainer = GNNLightningTrainer(training_manifest, model=mace_model)
trainer.train()
trainer.save_kim_model()
