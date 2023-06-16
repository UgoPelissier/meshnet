from typing import Tuple
import numpy as np
import os
import os.path as osp
import logging
from lightning.fabric.utilities.cloud_io import get_filesystem


def train_val_test_split(
        path: str,
        n: int,
        val_size: float,
        test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, validation and test sets."""
    indices = np.random.permutation(n)
    train_index, val_index, test_index = indices[:int(n*(1-(val_size+test_size)))], indices[int(n*(1-(val_size+test_size))):int(n*(1-test_size))],  indices[int(n*(1-test_size)):]
    np.savetxt(osp.join(path, 'data', 'indices', 'train_index.txt'), train_index, fmt='%i')
    np.savetxt(osp.join(path, 'data', 'indices', 'val_index.txt'), val_index, fmt='%i')
    np.savetxt(osp.join(path, 'data', 'indices', 'test_index.txt'), test_index, fmt='%i')
    return train_index, val_index, test_index


def load_train_val_test_index(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the train, validation and test sets indices."""
    return np.loadtxt(osp.join(path, 'data', 'indices', 'train_index.txt'), dtype=int), np.loadtxt(osp.join(path, 'data', 'indices', 'val_index.txt'), dtype=int), np.loadtxt(osp.join(path, 'data', 'indices', 'test_index.txt'), dtype=int)


def get_next_version(path: str) -> int:
    """Get the next version number for the logger."""
    log = logging.getLogger(__name__)
    fs = get_filesystem(path)

    try:
        listdir_info = fs.listdir(path)
    except OSError:
        log.warning("Missing logger folder: %s", path)
        return 0

    existing_versions = []
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if fs.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1