import os
import cv2
import numpy as np
from tqdm import  tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

__author__ = '__Girish_Hegde__'


class ImageSet(Dataset):
    def __init__(self, directory, ext=None, scale=1):
        super().__init__()
        self.imgs = [
            os.path.join(directory, name) 
            for name in os.listdir(directory) 
                if (ext is None) or name.endswith(ext)
        ]
        temp = []
        print('Reading Images ...')
        for filename in tqdm(self.imgs):
            img = cv2.imread(filename)
            if img.shape[0] != img.shape[1]: continue
            if scale < 1:
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            temp.append(torch.FloatTensor(img))

        print('Converting Images to torch ...')
        self.imgs = torch.stack(temp)
        self.imgs = ((self.imgs/255) - 0.5)/0.5
        self.imgs = self.imgs.permute(0, 3, 1, 2)

        print('Dataset Loaded Successfully') 
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


# https://github.com/MishaLaskin/vqvae/blob/master/utils.py
def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


# https://github.com/MishaLaskin/vqvae/blob/master/utils.py
def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader