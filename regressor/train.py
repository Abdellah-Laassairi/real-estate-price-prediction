from lightning import Trainer
from lightning.pytorch.cli import LightningCLI

from models.model import *
from pre.images import *

if __name__ == '__main__':
    LightningCLI(
        model_class=RealModel,
        datamodule_class=RealDataModule,
        trainer_class=Trainer,
        run=True,
    )
