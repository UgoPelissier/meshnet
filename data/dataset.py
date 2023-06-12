import os.path as osp
from typing import Optional, Callable
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data


class FreeFem(InMemoryDataset):
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
        if (self.split=='train'):
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif (self.split=='validation'):
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif (self.split=='test'):
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise ValueError("split must be 'train', 'validation' or 'test'")

    @property
    def raw_file_names(self) -> list:
        return ["stokes_{:03d}".format(i) for i in self.idx]

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "validation.pt", "test.pt"]

    def download(self):
        pass

    def line(
            self,
            start: np.ndarray,
            end: np.ndarray,
            scale: float
    ) -> float:
        return scale*float(np.linalg.norm(end-start))
    
    def ellipse(
            self,
            r1: float,
            r2: float,
            scale: float
    ) -> float:
        return scale*np.sqrt((r1**2 + r2**2)/2)

    def length(
            self,
            df: pd.DataFrame
    ) -> np.ndarray:
        length = []
        for i in range(df.shape[0]):
            temp = df.iloc[i]
            if (temp['type'] == 1):
                length.append(self.line(np.array(temp[['xstart', 'ystart', 'zstart']].values), np.array(temp[['xend', 'yend', 'zend']].values), temp['tend']-temp['tstart']))
            elif (temp['type'] == 2):
                length.append(self.ellipse(temp['radius1'], temp['radius2'], temp['tend']-temp['tstart']))
        return np.array(length)

    def process_file(
            self,
            name: str
    ) -> Data:
        df = pd.read_csv(osp.join(f'{self.raw_dir}/data/{name}.txt'), sep='\t')
        df['length'] = self.length(df)
        df['orientation'] = np.sign(df['n'])
        
        x = torch.tensor(np.array(df.drop(columns=['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend', 'label', 'n'])), dtype=torch.float32)
        pos = torch.tensor(df[['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend']].values)
        y = torch.abs(torch.tensor(df['length']/df['n'], dtype=torch.float32))
        name = f'{name}.txt'

        return Data(x=x, pos=pos, y=y, name=name)

    def process(self) -> None:
        data_list = []
        for name in self.raw_file_names:

            data = self.process_file(name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{self.split}.pt'))