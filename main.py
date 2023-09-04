from meshnet.data.datamodule import CadDataModule
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
                "save_dir": "/home/eleve05/adaptnet/meshnet/", # Parent directory of logs folder
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                "data.data_dir": "/home/eleve05/adaptnet/meshnet/data/stokes/", # Data directory

                "data.val_size": 0.15, # Validation size
                "data.test_size": 0.1, # Test size
                "data.batch_size": 1, # Batch size
                "data.num_workers": 4, # Number of workers

                "model.wdir": "/home/eleve05/adaptnet/meshnet/", # Working directory
                "model.data_dir": "/home/eleve05/adaptnet/meshnet/data/stokes/", # Data directory
                "model.logs": "/home/eleve05/adaptnet/meshnet/logs/", # Logs directory
                "model.dim": 2, # Dimension of the problem
                "model.num_layers": 15, # Number of layers
                "model.input_dim_node": 5, # Input dimension of the node features
                "model.input_dim_edge": 8, # Input dimension of the edge features
                "model.hidden_dim": 128, # Hidden dimension
                "model.output_dim": 1, # Output dimension
                "model.optimizer": "torch.optim.AdamW", # Optimizer

                "trainer.max_epochs": 100, # Maximum number of epochs
                "trainer.accelerator": "gpu", # Accelerator
                "trainer.devices": 1, # Number of devices
                "trainer.logger": logger, # Logger
                "trainer.callbacks": default_callbacks, # Callbacks
            },
        )

if __name__ == '__main__':
    cli = MyLightningCLI(
        model_class=MeshNet,
        datamodule_class=CadDataModule,
        seed_everything_default=42,
    )