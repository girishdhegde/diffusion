import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


__author__ = "__Girish_Hegde__"


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out or dim, 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim*4, dim_out or dim, 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    """ Positional Embedding

    Refs:
        https://huggingface.co/blog/annotated-diffusion
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """ Residual Block: conv2(scale_shit(conv(x), linear(time_emb))) + x
        author: girish d. hegde

    Refs:
        https://huggingface.co/blog/annotated-diffusion
    """
    def __init__(
        self,
        in_channels, out_channels,
        time_channels=None,
        groups=8,
    ):
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Sequential((
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        ))
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if time_channels is not None:
            self.time_emb = nn.Sequential((nn.SiLU(), nn.Linear(time_channels, out_channels*2)))
        else:
            self.time_emb = None

    def forward(self, x, time_emb=None):
        """
        Args:
            x (torch.tenosr): [b, c, h, w] - input features.
            time_emb (torch.tenosr): [b, t] - time embeddings.

        Returns:
            torch.tensor: [b, c_, h, w] - output features.
        """
        y = self.norm1(self.conv1(x))
        if (self.time_emb is not None) and (time_emb is not None):
            t = self.time_emb(time_emb)[..., None, None]  # [b, 2*c, 1, 1]
            scale, shift = t.chunk(2, dim=1)
            y = y*(scale + 1) + shift
        y = self.conv2(self.act1(y))
        return y + self.shortcut(x)


