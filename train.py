from lightning import Trainer
from lightning.pytorch.cli import LightningCLI

from models.model import *
from pre.images import *

TEST_SIZE = 0.2
SEED = 42

if __name__ == '__main__':

    # model = RealModel(

    # )
    # trainer = pl.Trainer(fast_dev_run=5,
    #                     devices=3,
    #                     accelerator="gpu",
    #                     train_dataloader=,
    #                     val_dataloader=,
    #                     )
    # trainer.fit(model)
    LightningCLI(model_class=RealModel,
                 datamodule_class=RealDataModule,
                 trainer_class=Trainer,
                 run=True)
