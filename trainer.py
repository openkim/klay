import os
import sys
import yaml
import numpy as np

import lightning as L
import torch
import torch_geometric as pyg
from torch_scatter import scatter
from simple_nequip import gen_model

from CoRe import CoRe

from torch_ema import ExponentialMovingAverage

torch.set_default_dtype(torch.float32)

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class PeriodicEpochCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    def __init__(self, every: int, checkpoint_dirpath: str = "manual_ckpts"):
        super().__init__()
        self.every = every
        self.ckpt_dirpath = checkpoint_dirpath

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0:
            os.makedirs(self.ckpt_dirpath, exist_ok=True)
            epoch_path = f"{self.ckpt_dirpath}/Epoch_{pl_module.current_epoch:06d}"
            os.makedirs(f"{epoch_path}", exist_ok=True)
            # Save the model
            if pl_module.ts: # if it is a torchscript model
                pl_module.model.save(f"{epoch_path}/model.pt")
            else: # if it is a yaml model
                torch.save(pl_module.model.state_dict(), f"{epoch_path}/model_state.pth")
            # Save the optimizer
            torch.save(pl_module.optimizer.state_dict(), f"{epoch_path}/optimizer_state.pth")


class LightningWrapper(L.LightningModule):
    def __init__(self, model, energy_weight=1.0, force_weight=10.0, batch_size=5, decay=0.99, ts=False, device="cuda"):
        super().__init__()
        if ts:
            self.model = model.float()
        else:
            # model is a input yaml file
            self.model = gen_model(model, save=False)
        print(type(self.model))
        self.ts = ts
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.val_step = 1
        self.batch_size = batch_size
        self.optimizer = None
        ema = ExponentialMovingAverage(self.model.parameters(), decay=decay)
        ema.to(device)
        self.ema = ema

    def forward(self, x, pos, edge_index, periodic_vec, batch_contrib):
        # pos.requires_grad_(True)
        E = self.model(x, pos, edge_index, periodic_vec, batch_contrib)
        F, = torch.autograd.grad([E], [pos], create_graph=True, grad_outputs=torch.ones_like(E))
        return E, -F

    def training_step(self, batch, batch_idx):
        x, pos, edge_index, periodic_vec, batch_contrib, E_target, F_target = batch.tags, batch.pos.requires_grad_(True), batch.edge_index, batch.periodic_vec, batch.batch, batch.y.float(), batch.force
        E, F = self.forward(x, pos, edge_index, periodic_vec, batch_contrib)
        loss = torch.nn.functional.mse_loss(E, E_target.view(-1,1)) * self.energy_weight 
        loss += torch.nn.functional.mse_loss(F, F_target) * self.force_weight
        self.log("train_loss", loss, on_epoch=True,batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # save as chkpoint
        torch.set_grad_enabled(True)
        x, pos, edge_index, periodic_vec, batch_contrib, E_target, F_target = batch.tags, batch.pos.requires_grad_(True), batch.edge_index, batch.periodic_vec, batch.batch, batch.y.float(), batch.force
        E, F = self.forward(x, pos, edge_index, periodic_vec, batch_contrib)
        eloss = torch.nn.functional.mse_loss(E, E_target.view(-1,1)) * self.energy_weight 
        floss = torch.nn.functional.mse_loss(F, F_target) * self.force_weight
        self.log("val_loss", eloss + floss, on_epoch=True,batch_size=self.batch_size)
        self.log("val_e_loss", eloss, on_epoch=True,batch_size=self.batch_size)
        self.log("val_f_loss", floss, on_epoch=True,batch_size=self.batch_size)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model.parameters())

#    def configure_optimizers(self):
#        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, amsgrad=True)
#        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
#                        patience=50,factor=0.5,)
#        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "monitor":"val_loss"}}

    def configure_optimizers(self):
        self.optimizer = CoRe(self.parameters(), lr=0.001)
        return self.optimizer # keeping everything default


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

    L.pytorch.seed_everything(42, workers=True)

    model_file = config["model_file"]
    if model_file[-3:] == ".pt":
        ts = True
    else:
        ts = False
    dataset_file = config["dataset_file"]
    indices_file = config["indices_file"]
    
    if ts:
        model = torch.jit.load(model_file)
        print(f"Loaded model: {model_file}")
    else:
        model = model_file # will be converted later

    indices = np.loadtxt(indices_file, dtype=int)
    train_indices = indices[:config["n_train"]]
    val_indices = indices[-config["n_val"]:]
    dataset = PYGLoad(path=dataset_file)
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    print(f"Loaded dataset: {dataset_file} \n Indices: {indices_file}")

    logger_csv = CSVLogger(f"lightning_logs/logs/CSV")
    logger_tb = TensorBoardLogger(save_dir=f"lightning_logs/logs/TB")
    
    #lightning_datamodule = pyg.data.lightning_datamodule.LightningDataset(train_dataset, val_dataset=val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"] )
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    val_loader = pyg.loader.DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    print("Initialized datamodule.")

    if config["device"] == "cuda" or config["device"] == "gpu":
        device = "cuda"
    else:
        device = "cpu"

    lightning_model = LightningWrapper(model, energy_weight=config["energy_weight"], force_weight=config["force_weight"], batch_size=config["batch_size"], device=device, ts=ts)

    print("Initialized model.")
    if "max_time" in config:
        max_time = config["max_time"]
    else:
        max_time = None

    if device != "cpu":
        trainer = L.Trainer(accelerator="gpu", devices=config["gpus"], max_epochs=config["max_epochs"],max_time=max_time,logger=[logger_tb, logger_csv],enable_checkpointing=True,callbacks=[PeriodicEpochCheckpoint(every=config["checkpoint_every"])])
    else:
        trainer = L.Trainer(max_epochs=config["max_epochs"], max_time=max_time,logger=[logger_tb, logger_csv],enable_checkpointing=True,deterministic=True,callbacks=[PeriodicEpochCheckpoint(every=config["checkpoint_every"])])
    print("Starting training.")
    #trainer.fit(lightning_model, train_loader, val_loader,ckpt_path="lightning_logs/logs/TB/lightning_logs/version_1/checkpoints/epoch=9-step=10.ckpt")
    trainer.fit(lightning_model, train_loader, val_loader)
    print("Done.")
