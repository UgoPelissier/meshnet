from typing import Optional
import os
import os.path as osp
import pygmsh
import gmsh

from meshnet.utils.stats import load_stats, normalize, unnormalize
from meshnet.utils.utils import get_next_version, generate_mesh_2d, generate_mesh_3d
from meshnet.model.processor import ProcessorLayer

import torch
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
            dim: int,
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
        self.data_dir = data_dir
        self.logs = logs
        self.dim = dim
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

        self.stats_loaded = False

    def build_processor_model(self):
        return ProcessorLayer

    def forward(
            self,
            batch: Data,
            split: str,
            mean_vec_x_predict: Optional[torch.Tensor] = None,
            mean_vec_edge_predict: Optional[torch.Tensor] = None,
            std_vec_x_predict: Optional[torch.Tensor] = None,
            std_vec_edge_predict: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr

        if split == 'train':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_train, self.mean_vec_edge_train], std=[self.std_vec_x_train, self.std_vec_edge_train])
        elif split == 'val':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_val, self.mean_vec_edge_val], std=[self.std_vec_x_val, self.std_vec_edge_val])
        elif split == 'test':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_test, self.mean_vec_edge_test], std=[self.std_vec_x_test, self.std_vec_edge_test])
        elif split == 'predict':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[mean_vec_x_predict, mean_vec_edge_predict], std=[std_vec_x_predict, std_vec_edge_predict]) # type: ignore
        else:
            raise ValueError(f'Invalid split: {split}')

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(self, pred: torch.Tensor, inputs: Data, split: str) -> torch.Tensor:
        """Calculate the loss for the given prediction and inputs."""
        # loss_mask = torch.argmax(inputs.x,dim=1)==torch.tensor(NodeType.OBSTACLE)

        # normalize labels with dataset statistics
        if split == 'train':
            labels = normalize(data=inputs.y, mean=self.mean_vec_y_train, std=self.std_vec_y_train)
        elif split == 'val':
            labels = normalize(data=inputs.y, mean=self.mean_vec_y_val, std=self.std_vec_y_val)
        elif split == 'test':
            labels = inputs.y
        else:
            raise ValueError(f'Invalid split: {split}')

        # find sum of square errors
        error = torch.sum((labels-pred.squeeze())**2)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error))
        
        return loss
    
    def on_test_start(self):
        """Set up folders for validation and test sets"""
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "vtk"), exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "msh"), exist_ok=True)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        pred = self(batch, split='train')
        loss = self.loss(pred, batch, split='train')
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #, batch_size=batch.x.shape[0])
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True) #, batch_size=batch.x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        if self.trainer.sanity_checking:
            self.load_stats()
        pred = self(batch, split='val')
        loss = self.loss(pred, batch, split='val')
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])
        return loss
    
    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        if not self.stats_loaded:
            self.load_stats()
            self.stats_loaded = True

        pred = unnormalize(
            data=self(batch, split='train'),
            mean=self.mean_vec_y_train,
            std=self.std_vec_y_train
        )

        loss = self.loss(pred, batch, split='test')
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.x.shape[0])

        if (self.dim==2):
            generate_mesh_2d(
                cad_path = osp.join(self.data_dir, 'raw', 'cad_{:03d}.geo'.format(batch.name[0])),
                batch=batch,
                pred=pred,
                save_dir=self.test_folder
            )
        elif (self.dim==3):
            generate_mesh_3d(
                cad_path = osp.join(self.data_dir, 'raw', 'cad_{:03d}.geo'.format(batch.name[0])),
                batch=batch,
                pred=pred,
                save_dir=self.test_folder
            )
        else:
            raise ValueError('Dimension must be 2 or 3')

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        
    def load_stats(self):
        """Load statistics from the dataset."""
        train_stats, val_stats, test_stats = load_stats(self.data_dir, self.device)
        self.mean_vec_x_train, self.std_vec_x_train, self.mean_vec_edge_train, self.std_vec_edge_train, self.mean_vec_y_train, self.std_vec_y_train = train_stats
        self.mean_vec_x_val, self.std_vec_x_val, self.mean_vec_edge_val, self.std_vec_edge_val, self.mean_vec_y_val, self.std_vec_y_val = val_stats
        self.mean_vec_x_test, self.std_vec_x_test, self.mean_vec_edge_test, self.std_vec_edge_test, self.mean_vec_y_test, self.std_vec_y_test = test_stats