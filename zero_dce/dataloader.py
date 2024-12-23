import os
import torch
import torchvision
import numpy as np
from random import shuffle
from torch.utils.data import Dataset
from PIL import Image
from .utils import add_noise

class SICEDataset(Dataset):

    def __init__(self, img_dir='.data/Dataset_Part1', img_files=None, image_size=256, transform=None):
        self.img_files = img_files
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform or torchvision.transforms.Compose([
            # Resize the image to 256x256
            torchvision.transforms.Resize((image_size, image_size)),
            # Convert the image (H x W x C) to a PyTorch tensor of shape (C x H x W) in range [0, 1]
            torchvision.transforms.ToTensor()
        ])

        # Shuffle the image files
        shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_files[item])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        if self.transform:
            img = self.transform(img)

        return img

class NoisyImageDataset(Dataset):

    def __init__(self, img_dir='.data/Dataset_Part1', img_files=None, image_size=256, noise_type='gaussian', noise_factor=0.1, transform=None):
        self.img_files = img_files
        self.img_dir = img_dir
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.transform = transform or torchvision.transforms.Compose([
            # Resize the image to 256x256
            torchvision.transforms.Resize((image_size, image_size)),
            # Convert the image (H x W x C) to a PyTorch tensor of shape (C x H x W) in range [0, 1]
            torchvision.transforms.ToTensor()
        ])

        # Shuffle the image files
        shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_files[item])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        noisy_img = add_noise(img, noise_type=self.noise_type, noise_factor=self.noise_factor)
        img = self.transform(img)

        return noisy_img, img