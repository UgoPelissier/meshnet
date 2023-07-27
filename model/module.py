from typing import Optional
import os
import os.path as osp

from meshnet.utils.utils import get_next_version
from meshnet.data.dataset import NodeType
from graphnet.model.processor import ProcessorLayer
from meshnet.model.mesh import post_process

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.data import Data
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


class MeshNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            wdir: str,
            data_dir: str,
            logs: str,
            num_layers: int,
            input_dim_node: int,
            input_dim_edge: int,
            hidden_dim: int,
            output_dim: int,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()

        self.wdir = wdir
        self.dataset = data_dir
        self.logs = logs
        self.num_layers = num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))


        self.processor = torch.nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, output_dim))
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.version = f'version_{get_next_version(path=self.logs)}'
        self.val_folder = osp.join(self.logs, self.version, 'val')
        self.test_folder = osp.join(self.logs, self.version, 'test')

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, batch: Data) -> torch.Tensor:
        """Forward pass of the model."""
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(self, pred: torch.Tensor, inputs: Data) -> torch.Tensor:
        """Calculate the loss for the given prediction and inputs."""
        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask = (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(NodeType.NORMAL)) + (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(NodeType.OUTFLOW)) + (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(NodeType.OBSTACLE))    

        # find sum of square errors
        error = torch.sum((inputs.y-pred)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss
    
    def on_test_start(self):
        """Set up folders for validation and test sets"""
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "mesh"), exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "vtu"), exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "tmp"), exist_ok=True)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        pred = self(batch)
        loss = loss = self.loss(pred, batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.x.shape[0])
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        pred = self(batch)
        loss = self.loss(pred, batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        return loss
    
    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        preds = self(batch)
        loss = self.loss(preds, batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        for pred, x, pos, name in zip(preds.split(sizes), batch.x.split(sizes), batch.pos.split(sizes), batch.name.split([1]*batch.name.shape[0])):
            for sample in self.test_idx:
                if (name==sample):
                    post_process(pred, x, pos, "stokes_{:03d}".format(name.item()), self.wdir, self.test_folder)

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]