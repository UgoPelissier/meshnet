from typing import Optional
import os
import os.path as osp
import shutil

from utils.utils import train_val_test_split, get_next_version
from model.mesh import post_process

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


class LightningNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            input_channels: int,
            wdir: str,
            data_dir: str,
            logs: str,
            val_size: float,
            test_size: float,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()
        
        # Define the model
        self.layer1 = torch.nn.Linear(input_channels, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 1)

        self.wdir = wdir
        self.dataset = data_dir
        self.logs = logs

        self.version = f'version_{get_next_version(path=self.logs)}'
        self.val_folder = osp.join(self.logs, self.version, 'val')
        self.test_folder = osp.join(self.logs, self.version, 'test')

        self.train_idx, self.val_idx, self.test_idx = train_val_test_split(path=data_dir, n=len(os.listdir(osp.join(data_dir, "raw", "cad"))), val_size=val_size, test_size=test_size)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch) -> torch.Tensor:
        """Forward pass of the model."""
        x = F.relu(self.layer1(batch.x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def on_test_start(self):
        """Set up folders for validation and test sets"""
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "mesh"), exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "tmp"), exist_ok=True)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), batch.name.split([1]*batch.name.shape[0])):
            for sample in self.test_idx:
                if (name==sample):
                    post_process(pred, x, pos, "stokes_{:03d}".format(name.item()), self.wdir, self.dataset, self.test_folder)

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]