from pathlib import Path
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
        net, itr, val_loss, train_loss, best, filename, **kwargs,
    ):
    ckpt = {
        'net': net,
        'training':{
            'iteration':itr, 'val_loss':val_loss, 'train_loss':train_loss, 'best':best,
        },
        'kwargs':kwargs,
    }
    torch.save(ckpt, filename)
    return ckpt


def load_checkpoint(filename):
    itr, best = 1, float('inf')
    net_ckpt, kwargs = None, None
    if filename is not None:
        if Path(filename).is_file():
            ckpt = torch.load(filename, map_location='cpu')
            net_ckpt = ckpt['net']
            if 'training' in ckpt:
                itr, val_loss, train_loss, best = ckpt['training'].values()
                print('Training parameters loaded successfully ...')
            if 'kwargs' in ckpt:
                kwargs = ckpt['kwargs']
                print('Additional kwargs loaded successfully ...')
    return net_ckpt, itr, best, kwargs


@torch.no_grad()
def write_pred(pred, outdir, name):
    outdir = Path(outdir)/name
    outdir.mkdir(exist_ok=True, parents=True)
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
    pred = (pred*0.5) + 0.5
    pred = (np.clip(pred, 0, 1)*255).astype(np.uint8)
    for i, img in enumerate(pred):
        filename = outdir/f'{i}.png'
        cv2.imwrite(str(filename), img[..., ::-1])
    return


class LossManager:
    def __init__(self, *losses, metric=None, best=float('inf')):
        self.accumulator = {name: 0. for name in losses}
        self.losses = {name: 0. for name in losses}
        self.steps = {name: 0 for name in losses}
        self.metric = metric if metric is not None else names[0]
        self.best = best

    def accumulate(self, steps=1, **kwargs):
        for k, v in kwargs.items():
            self.accumulator[k] += v
            self.steps[k] += 1
    
    def clear(self, *args):
        for k in args:
            self.accumulator[k] = 0
            self.steps[k] = 0

    def average(self, *args, clear=True):
        out = []
        for k in args:
            loss = self.accumulator[k]/self.steps[k]
            self.losses[k] = loss
            out.append(loss)
            if clear:
                self.accumulator[k] = 0.
                self.steps[k] = 0
        return out
    
    def update_best(self):
        if self.losses[self.metric] < self.best:
            self.best = self.losses[self.metric]
            return True
        return False

    def get_str(self, *args, spacer='\t'):
        return spacer.join(f'{k} = {self.losses[k]:.6f},' for k in args)

    def __str__(self):
        return '\t'.join(f'{k} = {v:.6f},' for k, v in self.losses.items()) + f'\tbest = {self.best:.6f}'

    def __repr__(self):
        return self.__str__()