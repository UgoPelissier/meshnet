from typing import Optional
import os
import os.path as osp
import pygmsh
import gmsh

from meshnet.utils.stats import load_stats, normalize, unnormalize
from meshnet.utils.utils import get_next_version
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
        os.makedirs(osp.join(self.test_folder, "msh"), exist_ok=True)
        os.makedirs(osp.join(self.test_folder, "vtk"), exist_ok=True)

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
        self.generate_mesh(
            cad_path = osp.join(self.data_dir, 'raw' , 'geo', 'cad_{:03d}.geo'.format(batch.name[0])),
            batch=batch,
            pred=pred,
            save_dir=self.test_folder
        )

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

    def generate_mesh(self, cad_path: str, batch: Data, pred: torch.Tensor, save_dir: str) -> None:
        with open (cad_path, 'r+') as f:
            # read the file
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

            # extract points, lines and circles
            points = [line for line in lines if line.startswith('Point')]
            lines__ = [line for line in lines if line.startswith('Line')]
            curve_loops = [line for line in lines if line.startswith('Curve Loop')]
            physical_curves = [line for line in lines if line.startswith('Physical Curve')]
            circles = [line for line in lines if line.startswith('Ellipse')]

            # extract coordinates and mesh size
            points_dict = {}
            for line in points:
                id = int(line.split('(')[1].split(')')[0])
                points_dict[id] = [float(p) for p in line.split('{')[1].split('}')[0].split(',')][:3]
            points_dict = dict(sorted(points_dict.items()))

            # extract edges
            lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
            circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
            edges = torch.cat([lines__, circles], dim=0)

            # list center points
            center_points = []
            for id in list(points_dict.keys()):
                if not id in edges:
                    center_points.append(id)

            # extract curve loops
            curve_loops_dict = {}
            for curve_loop in curve_loops:
                id = int(curve_loop.split('(')[1].split(')')[0])
                curve = [int(p) for p in curve_loop.split('{')[1].split('}')[0].split(',')]
                curve_loops_dict[id] = curve

            # Initialize empty geometry using the build in kernel in GMSH
            geometry = pygmsh.geo.Geometry()
            # Fetch model we would like to add data to
            model = geometry.__enter__()

            # Add points
            points_gmsh = []
            count = 0
            for id, point in points_dict.items():
                if id not in center_points:
                    points_gmsh.append(model.add_point(x=point, mesh_size=pred[(id-1)-count].cpu().item()))
                else:
                    points_gmsh.append(model.add_point(x=point, mesh_size=1.0))
                    count += 1

            # Add edges
            channnel_lines = []
            for edge in edges[:4,:]:
                channnel_lines.append(model.add_line(p0=points_gmsh[edge[0]-1], p1=points_gmsh[edge[1]-1]))

            start = 4
            while (start+4 <= len(points_dict)):
                channnel_lines.append(model.add_ellipse_arc(
                    start=points_gmsh[start+1],
                    center=points_gmsh[start],
                    point_on_major_axis=points_gmsh[start+2],
                    end=points_gmsh[start+2]
                ))
                channnel_lines.append(model.add_ellipse_arc(
                    start=points_gmsh[start+2],
                    center=points_gmsh[start],
                    point_on_major_axis=points_gmsh[start+3],
                    end=points_gmsh[start+3]
                ))
                channnel_lines.append(model.add_ellipse_arc(
                    start=points_gmsh[start+3],
                    center=points_gmsh[start],
                    point_on_major_axis=points_gmsh[start+1],
                    end=points_gmsh[start+1]
                ))
                start += 4

            # Add curve loops
            channel_loop = []
            for id, curve_loop in curve_loops_dict.items():
                channel_loop.append(model.add_curve_loop(curves=[channnel_lines[i-1] for i in curve_loop]))

            # Create a plane surface for meshing
            model.add_plane_surface(curve_loop=channel_loop[0], holes=channel_loop[1:])

            # Call gmsh kernel before add physical entities
            model.synchronize()

            geometry.generate_mesh(dim=2)
            
            gmsh.write(osp.join(save_dir, "msh", 'mesh_{:03d}.msh'.format(batch.name[0])))
            gmsh.write(osp.join(save_dir, "vtk", 'mesh_{:03d}.vtk'.format(batch.name[0])))
            
            gmsh.clear()
            geometry.__exit__()
    