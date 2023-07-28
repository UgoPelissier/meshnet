import os
import os.path as osp
import glob
from typing import Optional, Callable
import torch
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from torch_geometric.data import Dataset, Data
import enum

class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    INFLOW = 1
    OUTFLOW = 2
    WALL_BOUNDARY = 3
    OBSTACLE = 4
    SIZE = 5


class FreeFem(Dataset):
    """FreeFem dataset."""
    def __init__(
            self,
            root: str,
            split: str,
            idx: np.ndarray,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.split = split
        self.idx = idx
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list:
        return ["cad_{:03d}".format(i) for i in self.idx]

    @property
    def processed_file_names(self) -> list:
        return glob.glob(os.path.join(self.processed_dir, self.split, 'cad_*.pt'))

    def download(self):
        pass

    def node_type(self, label: str) -> int:
        if label == 'INFLOW':
            return NodeType.INFLOW
        elif label == 'OUTFLOW':
            return NodeType.OUTFLOW
        elif label == 'WALL_BOUNDARY':
            return NodeType.WALL_BOUNDARY
        elif label == 'OBSTACLE':
            return NodeType.OBSTACLE
        else:
            return NodeType.NORMAL

    def process_file(
            self,
            name: str
    ) -> None:
        with open(osp.join(self.raw_dir, 'geo', f'{name}.geo'), 'r') as f:
            # read lines and remove comments
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

            # extract points, lines and circles
            points = [line for line in lines if line.startswith('Point')]
            lines__ = [line for line in lines if line.startswith('Line')]
            physical_curves = [line for line in lines if line.startswith('Physical Curve')]
            circles = [line for line in lines if line.startswith('Ellipse')]

            # extract coordinates and mesh size
            points = torch.Tensor([[float(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in points])
            y = points[:, -1]
            points = points[:, :-1]

            # extract edges
            lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
            circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
            edges = torch.cat([lines__, circles], dim=0) -1

            count = 0
            for i in range(points.shape[0]):
                if not (i-count) in edges:
                    points = torch.cat([points[:i-count], points[i-count+1:]], dim=0)
                    edges = edges - 1*(edges>(i-count))
                    count += 1

            receivers = torch.min(edges, dim=1).values
            senders = torch.max(edges, dim=1).values
            packed_edges = torch.stack([senders, receivers], dim=1)
            # remove duplicates and unpack
            unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
            senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
            # create two-way connectivity
            edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

            # extract node types
            edge_types = torch.zeros(edges.shape[0], dtype=torch.long)
            for curve in physical_curves:
                label = curve.split('(')[1].split('"')[1]
                lines = curve.split('{')[1].split('}')[0].split(',')
                for line in lines:
                    edge_types[int(line)-1] = self.node_type(label)
            tmp = torch.zeros(edges.shape[0], dtype=torch.long)
            for i in range(len(permutation)):
                tmp[permutation[i]] = edge_types[i]
            edge_types = torch.cat((tmp, tmp), dim=0)
            edge_types_one_hot = torch.nn.functional.one_hot(edge_types.long(), num_classes=NodeType.SIZE)

            # get edge attributes
            u_i = points[edge_index[0]][:,:2]
            u_j = points[edge_index[1]][:,:2]
            u_ij = torch.Tensor(u_i - u_j)
            u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat((u_ij, u_ij_norm, edge_types_one_hot),dim=-1).type(torch.float)

            # get node attributes
            x = torch.zeros(points.shape[0], NodeType.SIZE)
            for i in range(edge_index.shape[0]):
                for j in range(edge_index.shape[1]):
                    x[edge_index[i,j], edge_types[j]] = 1.0

            torch.save(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name=torch.tensor(int(name[-3:]), dtype=torch.long)), osp.join(self.processed_dir, self.split, f'{name}.pt'))

    def process(self) -> None:
        """Process the dataset."""
        os.makedirs(os.path.join(self.processed_dir, self.split), exist_ok=True)
        print(f'{self.split} dataset')
        with alive_bar(total=len(self.processed_file_names)) as bar:
            for name in self.raw_file_names:
                self.process_file(name)
                bar()

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, self.split, "cad_{:03d}.pt".format(self.idx[idx])))
        return data