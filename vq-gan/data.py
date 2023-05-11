import os
import cv2
import numpy as np
from tqdm import  tqdm

import torch
from torch.utils.data import Dataset


__author__ = '__Girish_Hegde__'


class ImageSet(Dataset):
    def __init__(self, directory, ext=None):
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
