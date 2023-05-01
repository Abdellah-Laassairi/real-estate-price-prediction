from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.model import *
from pre.images import *

TEST_SIZE = 0.2
SEED = 42

if __name__ == '__main__':

    LightningCLI(model_class=Model,
                 datamodule_class=RealDataset,
                 trainer_class=Trainer,
                 save_config_overwrite=True,
                 run=True)
