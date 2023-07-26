import os
import os.path as osp
import glob
from typing import Optional, Callable
import torch
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from torch_geometric.data import Dataset, Data


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
        df = pd.read_csv(osp.join(f'{self.raw_dir}/cad/{name}.txt'), sep='\t')
        df['orientation'] = np.sign(df['n'])
        
        x = torch.tensor(np.array(df.drop(columns=['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend', 'label', 'n'])), dtype=torch.float32)
        pos = torch.tensor(df[['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend']].values)
        y = torch.abs(torch.tensor(df['length']/df['n'], dtype=torch.float32))

        torch.save(Data(x=x, pos=pos, y=y, name=torch.tensor(int(name[-3:]), dtype=torch.long)), osp.join(self.processed_dir, self.split, f'{name}.pt'))

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