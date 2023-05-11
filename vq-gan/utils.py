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
