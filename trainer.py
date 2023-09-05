import sys
import yaml
import numpy as np

import lightning as L
import torch
import torch_geometric as pyg
from torch_scatter import scatter

from CoRe import CoRe

torch.set_default_dtype(torch.float32)

class LightningWrapper(L.LightningModule):
    def __init__(self, model, energy_weight=1.0, force_weight=10.0):
        super().__init__()
        self.model = model.float()
        self.energy_weight = energy_weight
        self.force_weight = force_weight

    def forward(self, x, pos, edge_index, periodic_vec, batch_contrib):
        # pos.requires_grad_(True)
        E = self.model(x, pos, edge_index, periodic_vec, batch_contrib)
        F, = torch.autograd.grad([E], [pos], create_graph=True, grad_outputs=torch.ones_like(E))
        return E, F

    def training_step(self, batch, batch_idx):
        x, pos, edge_index, periodic_vec, batch_contrib, E_target, F_target = batch.tags, batch.pos.requires_grad_(True), batch.edge_index, batch.periodic_vec, batch.batch, batch.y.float(), batch.force
        E, F = self.forward(x, pos, edge_index, periodic_vec, batch_contrib)
        loss = torch.nn.functional.mse_loss(E, E_target.view(-1,1)) * self.energy_weight 
        loss += torch.nn.functional.mse_loss(F, F_target) * self.force_weight
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        x, pos, edge_index, periodic_vec, batch_contrib, E_target, F_target = batch.tags, batch.pos.requires_grad_(True), batch.edge_index, batch.periodic_vec, batch.batch, batch.y.float(), batch.force
        E, F = self.forward(x, pos, edge_index, periodic_vec, batch_contrib)
        eloss = torch.nn.functional.mse_loss(E, E_target.view(-1,1)) * self.energy_weight 
        floss = torch.nn.functional.mse_loss(F, F_target) * self.force_weight
        self.log("val_loss", eloss + floss, on_epoch=True)
        self.log("val_e_loss", eloss, on_epoch=True)
        self.log("val_f_loss", floss, on_epoch=True)


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                        patience=50,factor=0.5,)
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "monitor":"val_loss"}}

#    def configure_optimizers(self):
#        self.optimizer = CoRe(self.parameters)
#        return self.optimizer # keeping everything default


####################################################
# data
####################################################

class PYGLoad(pyg.data.InMemoryDataset):
    def __init__(self,transform=None, pre_transform=None, path="./data.pt"):
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = torch.load(path)


if __name__ == "__main__":
    print("Starting training script")
    if len(sys.argv) != 2:
        print("Usage: python trainer.py yaml_file")
        sys.exit(1)
    print(f"Config file: {sys.argv[1]}")
    yaml_file = sys.argv[1]
    config = yaml.safe_load(open(yaml_file))
    print(f"Config: {config}") 

    model_file = config["model_file"]
    dataset_file = config["dataset_file"]
    indices_file = config["indices_file"]

    model = torch.jit.load(model_file)
    print(f"Loaded model: {model_file}")

    indices = np.loadtxt(indices_file, dtype=int)
    train_indices = indices[:config["n_train"]]
    val_indices = indices[-config["n_val"]:]
    dataset = PYGLoad(path=dataset_file)
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    print(f"Loaded dataset: {dataset_file} \n Indices: {indices_file}")

    #lightning_datamodule = pyg.data.lightning_datamodule.LightningDataset(train_dataset, val_dataset=val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"] )
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    val_loader = pyg.loader.DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    print("Initialized datamodule.")

    lightning_model = LightningWrapper(model, energy_weight=config["energy_weight"], force_weight=config["force_weight"])
    print("Initialized model.")
    if "max_time" in config:
        max_time = config["max_time"]
    else:
        max_time = None

    if config["device"] == "gpu":
        trainer = L.Trainer(strategy="ddp", accelerator="gpu", devices=config["gpus"], max_epochs=config["max_epochs"],max_time=max_time)
    else:
        trainer = L.Trainer(max_epochs=config["max_epochs"], max_time=max_time)
    print("Starting training.")
    trainer.fit(lightning_model, train_loader, val_loader)
    print("Done.")
