from typing import Optional
import numpy as np
import os
import os.path as osp
import shutil

from utils.utils import load_train_val_test_index, get_next_version
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
            path: str,
            dataset: str,
            logs: str,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()
        
        # Define the model
        self.layer1 = torch.nn.Linear(input_channels, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 1)

        self.path = path
        self.dataset = dataset
        self.logs = logs

        self.version = f'version_{get_next_version(path=self.logs)}'
        self.val_folder = osp.join(self.logs, self.version, 'val')
        self.test_folder = osp.join(self.logs, self.version, 'test')

        _, self.val_idx, self.test_idx = load_train_val_test_index(path=self.path)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch) -> torch.Tensor:
        """Forward pass of the model."""
        x = F.relu(self.layer1(batch.x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def on_train_start(self):
        """Set up folders for validation and test sets"""
        if self.trainer.global_rank == 0:
            if not osp.exists(self.val_folder):
                os.makedirs(self.val_folder)
                os.makedirs(osp.join(self.val_folder, "pred"))
                os.makedirs(osp.join(self.val_folder, "error"))
                os.makedirs(osp.join(self.val_folder, "true"))
                os.makedirs(osp.join(self.val_folder, "tmp"))

                for sample in self.val_idx:
                    os.makedirs(osp.join(self.val_folder, "pred", f'stokes_{sample:03}'))
                    os.makedirs(osp.join(self.val_folder, "error", f'stokes_{sample:03}'))
                    shutil.copyfile(
                        osp.join(self.dataset, 'raw', 'sol', f'stokes_{sample:03}.vtu'),
                        osp.join(self.val_folder, "true", f'stokes_{sample:03}.vtu')
                    )

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

        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), batch.name.split([1]*batch.name.shape[0])):
            for sample in self.val_idx:
                if (name==sample):
                    post_process(pred, x, pos, "stokes_{:03d}".format(name.item()), self.path, self.dataset, self.val_folder, epoch=self.trainer.current_epoch)

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]