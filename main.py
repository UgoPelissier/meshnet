from typing import Union
from data.datamodule import FreeFemDataModule
from model.model import LightningNet

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import warnings

class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""
    warnings.filterwarnings("ignore")

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        default_callbacks = [
            {
                "class_path": "callbacks.modelsummary.MyRichModelSummary",
            },
            {
                "class_path": "callbacks.progressbar.MyProgressBar",
            }
        ]

        logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                # "save_dir": "/root/safran/data/", # Parent directory of logs folder
                "save_dir": "/data/users/upelissier/30-Code/meshnet/", # Parent directory of logs folder
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                # "data.path": "/root/safran/meshnet/", # Working directory
                # "data.dataset": "/root/safran/data/", # Data directory
                "data.path": "/home/upelissier/30-Code/meshnet/", # Working directory
                "data.dataset": "/data/users/upelissier/30-Code/freefem/", # Data directory

                "data.val_size": 0.1,
                "data.test_size": 0.15,
                "data.batch_size": 8,
                "data.num_workers": 4,

                "model.input_channels": 7,

                # "model.path": "/root/safran/meshnet/", # Working directory
                # "model.dataset": "/root/safran/data/", # Data directory
                # "model.logs": "/root/safran/data/logs/", # Logs directory
                "model.path": "/home/upelissier/30-Code/meshnet/", # Working directory
                "model.dataset": "/data/users/upelissier/30-Code/freefem/", # Data directory
                "model.logs": "/data/users/upelissier/30-Code/meshnet/logs/", # Logs directory

                "model.optimizer": "torch.optim.AdamW",

                "trainer.max_epochs": 30,
                "trainer.accelerator": "cpu",
                "trainer.devices": 1,
                "trainer.logger": logger,
                "trainer.callbacks": default_callbacks,
            },
        )

if __name__ == '__main__':
    cli = MyLightningCLI(
        model_class=LightningNet,
        datamodule_class=FreeFemDataModule,
        seed_everything_default=42,
    )