import numpy as np
import os
import os.path as osp
import shutil

from utils.utils import load_train_val_test_index, get_next_version
from model.loss import projection_loss

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


class LightningNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            input_channels: int = 7,
            path: str = '/home/upelissier/30-Implements/meshnet/',
            dataset: str = '/data/users/upelissier/30-Implements/freefem/',
            logs: str = '/data/users/upelissier/30-Implements/meshnet/logs/',
            version: str = None,
            val_idx: list = None,
            optimizer: OptimizerCallable = torch.optim.AdamW,
            lr_scheduler: LRSchedulerCallable = None
        ) -> None:
        super().__init__()
        
        # Define the model
        self.layer1 = torch.nn.Linear(input_channels, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 1)

        self.path = path
        self.dataset = dataset
        self.logs = logs

        if version is None:
            self.version = f'version_{get_next_version(path=self.logs)}'
        else:
            self.version = version

        self.val_folder = osp.join(self.logs, self.version, 'val')
        self.test_folder = osp.join(self.logs, self.version, 'test')

        if val_idx is None:
            _, self.val_idx, self.test_idx = load_train_val_test_index(path=self.path)
        else:
            self.val_idx = val_idx

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch) -> torch.Tensor:
        """Forward pass of the model."""
        x = F.relu(self.layer1(batch.x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def init_folder(self, folder: str) -> None:
        """Initialize the folder for the validation and test."""
        if not osp.exists(folder):
            os.makedirs(folder)
            os.makedirs(osp.join(folder, "pred"))
            os.makedirs(osp.join(folder, "true"))
            os.makedirs(osp.join(folder, "tmp"))

            for sample in self.val_idx:
                os.makedirs(osp.join(folder, "pred", f'stokes_{sample:03}'))
                shutil.copyfile(
                    osp.join(self.dataset, 'raw', 'sol', f'stokes_{sample:03}.vtu'),
                    osp.join(folder, "true", f'stokes_{sample:03}.vtu')
                )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        rank = self.trainer.global_rank
        if rank == 0:
            self.init_folder(folder=self.val_folder)

        loss_proj = 0
        preds = self(batch)
        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()

        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), np.array_split(np.array(batch.name), len(batch.name))):
                loss_proj += projection_loss(pred, x, pos, name[0], self.path, self.dataset, self.val_folder)
                    
        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])
        self.log("train/projection_loss", loss_proj, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        preds = self(batch)
        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        loss_proj = 0

        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), np.array_split(np.array(batch.name), len(batch.name))):
            for sample in self.val_idx:
                if ((int(name.item()[-7:-4])==sample)):
                    loss_proj += projection_loss(pred, x, pos, name[0], self.path, self.dataset, self.val_folder, test=True, epoch=self.trainer.current_epoch)

        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])
        self.log("val/projection_loss", loss_proj, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Test step of the model."""
        rank = self.trainer.global_rank
        if rank == 0:
            self.init_folder(folder=self.test_folder)

        preds = self(batch)
        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        loss_proj = 0

        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), np.array_split(np.array(batch.name), len(batch.name))):
            for sample in self.test_idx:
                if ((int(name.item()[-7:-4])==sample)):
                    loss_proj += projection_loss(pred, x, pos, name[0], self.path, self.dataset, self.test_folder, test=True, epoch=self.trainer.current_epoch)

        loss = F.mse_loss(preds, batch.y.unsqueeze(dim=-1))

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])
        self.log("test/projection_loss", loss_proj, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}