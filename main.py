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
                "save_dir": "/data/users/upelissier/30-Code/meshnet/", # Parent directory of logs folder
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                "data.data_dir": "/data/users/upelissier/30-Code/meshnet/data/stokes/", # Data directory

                "data.val_size": 0.15,
                "data.test_size": 0.1,
                "data.batch_size": 1,
                "data.num_workers": 4,

                "model.wdir": "/home/upelissier/30-Code/meshnet/", # Working directory
                "model.data_dir": "/data/users/upelissier/30-Code/meshnet/data/stokes/", # Data directory
                "model.logs": "/data/users/upelissier/30-Code/meshnet/logs/", # Logs directory
                "model.num_layers": 15,
                "model.input_dim_node": 5,
                "model.input_dim_edge": 8,
                "model.hidden_dim": 128,
                "model.output_dim": 1,
                "model.optimizer": "torch.optim.AdamW",

                "trainer.max_epochs": 100,
                "trainer.accelerator": "gpu",
                "trainer.devices": 1,
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