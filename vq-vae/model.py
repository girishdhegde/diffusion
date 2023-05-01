import torch
import torch.nn as nn
from einops import repeat, rearrange


__author__ = "__Girish_Hegde__"


class ResBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, hidden_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch - in_ch, 1, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class Encoder(nn.Module):
    """ Encoder: maps data 'x' to continous latent vectors 'ze'.
    """
    def __init__(self, in_ch=3, res_layers=2, hidden_ch=256, ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch // 2, 4, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch // 2, hidden_ch, 4, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
            *(ResBlock(hidden_ch, hidden_ch//2, hidden_ch) for _ in range(res_layers)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """ Encoder: maps nearest embedding vectors 'zq' to data/output 'x'.
    """
    def __init__(self, in_ch=3, res_layers=2, hidden_ch=256, ):
        super().__init__()
        self.layers = nn.Sequential(
            *(ResBlock(hidden_ch, hidden_ch//2, hidden_ch) for _ in range(res_layers)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, 1, 1),
        )


    def forward(self, x):
        return self.layers(x)
