# -*- coding: utf-8 -*-
"""dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FRghA-D1q8ChAIE18qV2MHUQnVgnTfO8
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SatelliteDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        """
        Args:
            lr_dir (string): Directory with all the low-resolution images.
            hr_dir (string): Directory with all the high-resolution images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            Tuple (Tensor, Tensor): Low-resolution image and high-resolution image.
        """
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

def get_data_loader(lr_dir, hr_dir, batch_size=16, image_size=128):
    """
    Args:
        lr_dir (string): Directory with all the low-resolution images.
        hr_dir (string): Directory with all the high-resolution images.
        batch_size (int): Number of samples per batch to load.
        image_size (int): Size to which the images will be resized.
    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    dataset = SatelliteDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return data_loader

# Example usage
if __name__ == "__main__":
    lr_dir = "D:/download/StageRecherche/lr_sentinel-1"
    hr_dir = "D:/download/StageRecherche/hr_venus"

    data_loader = get_data_loader(lr_dir, hr_dir, batch_size=16, image_size=128)

    # Iterate through the dataset
    for lr_image, hr_image in data_loader:
        print(f"Low-resolution image shape: {lr_image.shape}, High-resolution image shape: {hr_image.shape}")