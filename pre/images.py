import glob
import math
import os
from pathlib import Path
from typing import Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyparsing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import transforms
# PL

# Transforms
DATA_TRANSFORMS = {
    'train':
    transforms.Compose([
        #transforms.Pad(padding, fill=0, padding_mode='constant'),
        transforms.Resize((342, 460)),  # 224   342 460
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize((342, 460)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class RealDataset(Dataset):
    """Real dataset."""
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.target = self.df.price
        self.__quantize__()  # Quantize targets for classification

        self.root_dir = root_dir
        self.transform = transform

    def __quantize__(self):
        self.target_quantized = self.target // 100
        self.target_quantized = self.target_quantized.astype('int32')
        self.target_classes, self.target_price = pd.factorize(
            self.target_quantized)
        self.target_price = self.target_price.to_numpy()
        self.target_log = np.log(self.target)

    def resize_images(self, images):
        # Calculate the average width and height
        total_width = 0
        total_height = 0
        num_images = len(images)
        for img in images:
            total_width += img.width
            total_height += img.height
        avg_width = total_width // num_images
        avg_height = total_height // num_images
        # Resize all the images to the average size
        resized_images = []
        for img in images:
            resized_img = img.resize((avg_width, avg_height))
            resized_images.append(resized_img)

        return resized_images

    def __stack__(self, id):
        full_path = os.path.join(self.root_dir, f'ann_{id}')
        images_paths = glob.glob(f'{full_path}/*.jpg')
        images = [Image.open(i) for i in images_paths]

        new_images = self.resize_images(images)

        # Get dimensions of all images
        heights = [img.size[1] for img in new_images]
        widths = [img.size[0] for img in new_images]

        # Find the maximum height and width among all images
        n_images = len(images)
        n_rows = n_images // 2 + (n_images % 2 > 0)
        max_height = max(heights)
        current_width = 0
        for i in range(n_rows):
            width, height = new_images[i].size
            current_width += width
        max_width = current_width

        # Create a new image with the maximum height and total width
        stacked_image = Image.new('RGB', (max_width, max_height * 2))

        # Iterate over each image and copy it to the stacked image on a new column
        current_width = 0
        current_width = 0
        for i in range(n_rows):
            width, height = new_images[i].size
            stacked_image.paste(new_images[i], (current_width, 0))
            current_width += width

        current_width = 0
        for j in range(n_rows, len(images)):
            width, height = new_images[i].size
            stacked_image.paste(new_images[j], (current_width, max_height))
            current_width += width

        return stacked_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_annonce = int(self.df.iloc[idx]['id_annonce'])
        image = self.__stack__(id_annonce)
        price = self.target_log[idx]

        if self.transform:
            image = self.transform(image)

        return image, price


class RealDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_src,
                 cache_path: Union[str, Path],
                 batch_size: int = 10,
                 reuse_cache: bool = False,
                 nworkers: int = 20,
                 test_size: float = 0.1,
                 seed: int = 42,
                 verbose: bool = True):
        super().__init__()

        self.train_src = train_src
        self.cache_path = cache_path
        self.reuse_cache = reuse_cache
        self.nworkers = nworkers
        self.verbose = verbose
        self.nclasses = self.mapping.nclasses
        self.seed = seed
        self.test_size = test_size

        self.common_params = dict(cache_name=None,
                                  nworkers=self.nworkers,
                                  verbose=self.verbose)

        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.cache_paths = {
            'full': self.cache_path / 'full.ckpt',
        }

    def _prepare_dataset(self,
                         img_sources,
                         cache_path: Union[Path, str],
                         nworkloads: int = 1):

        if self.reuse_cache and Path(cache_path).is_file():
            print(f'Reusing train cache in file {cache_path}')
        else:

            full_dataset_ds = RealDataset(img_sources,
                                          DATA_TRANSFORMS['train'])

            assert nworkloads > 0

            full_dataset_ds.create_n_workloads(nworkloads)

            print('[bold cyan]Exporting dataset...')
            full_dataset_ds.export(filename=cache_path)

    def prepare_data(self):

        nworkloads = 1
        if self.trainer is not None:
            max_epochs = self.trainer.max_epochs or 1
            reload_frequency = self.trainer.reload_dataloaders_every_n_epochs
            if reload_frequency not in {None, 0}:
                nworkloads = math.ceil(max_epochs / reload_frequency)

        logger.info('Creating %d train workloads', nworkloads)

        console.rule('[bold cyan]Preparing Real estat Dataset')

        self._prepare_dataset(
            self.train_src,
            self.cache_paths['full'],
            nworkloads=nworkloads,
        )

    def setup(self, stage: str):
        # This runs across all GPUs and it is safe to make state assignments here

        # # stages "fit", "predict", "test"

        if stage == 'fit':
            self.full_ds: RealDataset = torch.load(self.cache_paths['train'])

            # generate indices: instead of the actual data we pass in integers instead
            train_indices, test_indices, _, _ = train_test_split(
                range(len(self.full_ds)),
                self.full_ds.target,
                test_size=self.test_size,
                random_state=self.seed)

            self.ds_train = Subset(self.full_ds, train_indices)
            self.ds_val = Subset(self.full_ds, test_indices)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          num_workers=self.nworkers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=self.batch_size,
                          num_workers=self.nworkers,
                          shuffle=False,
                          pin_memory=True)
