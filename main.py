from data.datamodule import FreeFemDataModule
from model.model import LightningNet

from lightning.pytorch.cli import LightningCLI
import warnings
from lightning.pytorch.callbacks import RichModelSummary

class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""
    warnings.filterwarnings("ignore")

    def add_arguments_to_parser(self, parser):
        """Add arguments to parser."""
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
                "save_dir": "/data/users/upelissier/30-Implements/meshnet/",
                "name": "logs/",
            },
        }
        parser.set_defaults(
            {
                "trainer.max_epochs": 30,
                "trainer.accelerator": "gpu",
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