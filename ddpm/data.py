import re
import random
from pathlib import Path
from copy import deepcopy
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


__author__ = "__Girish_Hegde__"


class ImageSet(Dataset):
    """ Pytorch Dataset class for Images.

    Args:
        size (int): image size.
        split (string): 'train' or 'val' or 'test'.
    """
    def __init__(self, size=64, split='train'):
        super().__init__()
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split=split)
        self.ds = [sample["image"].resize((size, size)) for sample in ds]
        self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.len = len(self.ds)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
            torch.FloatTensor: [c, h, w] - ouput image.
        """
        return self.transform(self.ds[index])