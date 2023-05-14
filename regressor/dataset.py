import glob
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
        transforms.Resize((342, 460)),
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
