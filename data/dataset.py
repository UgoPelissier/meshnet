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
            circles = [line for line in lines if line.startswith('Circle')]

            # extract coordinates and mesh size
            points = torch.Tensor([[float(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in points])
            y = points[:, -1]
            pos = points[:, :-1]

            # extract edges
            lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
            circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
            edges = torch.cat([lines__, circles], dim=0)
            receivers = torch.min(edges, dim=1).values
            senders = torch.max(edges, dim=1).values
            packed_edges = torch.stack([senders, receivers], dim=1)
            # remove duplicates and unpack
            unique_edges = torch.unique(packed_edges, dim=0)
            senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
            # create two-way connectivity
            edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

            torch.save(Data(x=x, edge_index=edge_index, pos=pos, y=y, name=torch.tensor(int(name[-3:]), dtype=torch.long)), osp.join(self.processed_dir, self.split, f'{name}.pt'))

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
        data = torch.load(os.path.join(self.processed_dir, self.split, "stokes_{:03d}.pt".format(self.idx[idx])))
        return data