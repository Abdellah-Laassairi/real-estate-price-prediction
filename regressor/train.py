from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from loguru import logger as log

from models.model import *
from pre.images import *

if __name__ == '__main__':
    log.info('Starting Lightning CLI training')
    LightningCLI(
        model_class=RealModel,
        datamodule_class=RealDataModule,
        trainer_class=Trainer,
        run=True,
    )
