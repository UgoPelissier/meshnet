from meshnet.data.datamodule import FreeFemDataModule
from meshnet.model.module import MeshNet

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
                # "save_dir": "/home/eleve05/safran/meshnet/", # Parent directory of logs folder
                "save_dir": "/data/users/upelissier/30-Code/meshnet/", # Parent directory of logs folder
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                # "data.data_dir": "/home/eleve05/safran/meshnet/data", # Data directory
                "data.data_dir": "/data/users/upelissier/30-Code/meshnet/data/stokes/", # Data directory

                "data.val_size": 0.1,
                "data.test_size": 0.15,
                "data.batch_size": 8,
                "data.num_workers": 4,

                "model.input_channels": 7,

                # "model.wdir": "/home/eleve05/safran/meshnet/", # Working directory
                # "model.data_dir": "/home/eleve05/safran/meshnet/data/", # Data directory
                # "model.logs": "/home/eleve05/safran/meshnet/logs/", # Logs directory
                "model.wdir": "/home/upelissier/30-Code/meshnet/", # Working directory
                "model.data_dir": "/data/users/upelissier/30-Code/meshnet/data/stokes/", # Data directory
                "model.logs": "/data/users/upelissier/30-Code/meshnet/logs/", # Logs directory
                "model.val_size": 0.15,
                "model.test_size": 0.1,
                "model.optimizer": "torch.optim.AdamW",

                "trainer.max_epochs": 100,
                "trainer.accelerator": "gpu",
                "trainer.devices": 2,
                "trainer.logger": logger,
                "trainer.callbacks": default_callbacks,
            },
        )

if __name__ == '__main__':
    cli = MyLightningCLI(
        model_class=MeshNet,
        datamodule_class=FreeFemDataModule,
        seed_everything_default=42,
    )