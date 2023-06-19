from typing import Union
from data.datamodule import FreeFemDataModule
from model.model import LightningNet

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import warnings

class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""
    warnings.filterwarnings("ignore")

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--working_dir",
            type=str,
            default="/root/safran/meshnet/", # "/home/upelissier/30-Implements/meshnet/"
            help=(
                "Working directory"
            ),
        )

        parser.add_argument(
            "--data_dir",
            type=str,
            default="/root/safran/data/", # "/data/users/upelissier/30-Implements/freefem/" 
            help=(
                "Data directory"
            ),
        )

        """Add arguments to parser."""
        parser.add_argument(
            "--parent_logger_save_dir",
            type=str,
            default="/root/safran/data/", # "/data/users/upelissier/30-Implements/meshnet/"
            help=(
                "Parent directory of logs folder"
            ),
        )

        """Add arguments to parser."""
        parser.add_argument(
            "--logger_save_dir",
            type=str,
            default="/root/safran/data/logs/", # "/data/users/upelissier/30-Implements/meshnet/logs/"
            help=(
                "Logs directory"
            ),
        )

        default_callbacks = [
            {
                "class_path": "utils.modelsummary.MyRichModelSummary",
            },
            {
                "class_path": "utils.progressbar.MyProgressBar",
            }
        ]

        logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": "/root/safran/data/", # "/data/users/upelissier/30-Implements/meshnet/" # Parent directory of logs folder
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                "data.path": "/root/safran/meshnet/", # "/home/upelissier/30-Implements/meshnet/" # Working directory
                "data.dataset": "/root/safran/data/", # "/data/users/upelissier/30-Implements/freefem/" # Data directory
                "data.val_size": 0.1,
                "data.test_size": 0.15,
                "data.batch_size": 8,
                "data.num_workers": 4,

                "model.input_channels": 7,
                "model.path": "/root/safran/meshnet/", # "/home/upelissier/30-Implements/meshnet/" # Working directory
                "model.dataset": "/root/safran/data/", # "/data/users/upelissier/30-Implements/freefem/" # Data directory
                "model.logs": "/root/safran/data/logs/", # "/data/users/upelissier/30-Implements/meshnet/logs/" # Logs directory
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