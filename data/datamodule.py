import os
import os.path as osp

from meshnet.utils.utils import train_val_test_split
from meshnet.data.dataset import FreeFem

from torch_geometric.loader import DataLoader
import lightning.pytorch as pl


class FreeFemDataModule(pl.LightningDataModule):
    """Lightning data module for the FreeFem dataset."""
    def __init__(
            self,
            data_dir: str,
            val_size: float,
            test_size: float,
            batch_size: int,
            num_workers: int
        ) -> None:
        super().__init__()
        # Define the indices
        self.train_idx, self.val_idx, self.test_idx = train_val_test_split(path=data_dir, n=len(os.listdir(osp.join(data_dir, "raw", "geo"))), val_size=val_size, test_size=test_size)

        # Define the dataset
        self.train_dataset = FreeFem(root=data_dir, split='train', idx=self.train_idx)
        self.val_dataset = FreeFem(root=data_dir, split='validation', idx=self.val_idx)
        self.test_dataset = FreeFem(root=data_dir, split='test', idx=self.test_idx)

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