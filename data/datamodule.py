import os
import os.path as osp

from utils.utils import train_val_test_split
from data.dataset import FreeFem

from torch_geometric.loader import DataLoader
import lightning.pytorch as pl


class FreeFemDataModule(pl.LightningDataModule):
    """Lightning data module for the FreeFem dataset."""
    def __init__(
            self,
            path: str = '/home/upelissier/30-Implements/meshnet/',
            dataset: str = '/data/users/upelissier/30-Implements/freefem/',
            val_size: float = 0.1,
            test_size: float = 0.15,
            batch_size: int = 8,
            num_workers: int = 4
        ) -> None:
        super().__init__()
        # Define the indices
        self.train_index, self.val_index, self.test_index = train_val_test_split(path=path, n=len(os.listdir(osp.join(dataset, "raw", "data"))), val_size=val_size, test_size=test_size)

        # Define the dataset
        self.train_dataset = FreeFem(root=dataset, split='train', idx=self.train_index)
        self.val_dataset = FreeFem(root=dataset, split='validation', idx=self.val_index)
        self.test_dataset = FreeFem(root=dataset, split='test', idx=self.test_index)

        # Define the parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)