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


class DiffusionSet(Dataset):
    """ Pytorch Dataset class for Images.

    Args:
        size (int): image size.
        split (string): 'train' or 'val' or 'test'.
        timesteps (int): diffusion timesteps.
    """
    def __init__(self, size=64, split='train', timesteps=100):
        super().__init__()
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split=split)
        self.ds = [sample["image"].resize((size, size)) for sample in ds]
        self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.t = timesteps - 1
        self.len = len(self.ds)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
            torch.FloatTensor: [c, h, w] - ouput image.
            torch.LongTensor: [1, ] - timestep.
        """
        t = random.randint(0, self.t)
        return self.transform(self.ds[index]), torch.tensor([t], dtype=torch.int64)