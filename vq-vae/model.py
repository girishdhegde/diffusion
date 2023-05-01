import torch
import torch.nn as nn
from einops import repeat, rearrange


__author__ = "__Girish_Hegde__"


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1, 1),
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
            nn.ReLU(),
            nn.Conv2d(hidden_ch // 2, hidden_ch, 4, 2, padding=1),
            nn.ReLU(),
            *(ResBlock(hidden_ch) for _ in range(res_layers)),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """ Encoder: maps nearest embedding vectors 'zq' to data/output 'x'.
    """
    def __init__(self, in_ch=3, res_layers=2, hidden_ch=256, ):
        super().__init__()
        self.layers = nn.Sequential(
            *(ResBlock(hidden_ch) for _ in range(res_layers)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class VectorQuantizer(nn.Module):
    """ VQ: converts continous latents 'ze' to discrete latents 'z' then maps 'z' to nearest embedding vectors 'zq'. 
    """
    def __init__(self, num_emb, dimension):
        super().__init__()
        self.code_book = nn.Parameter(torch.FloatTensor(num_emb, dimension).uniform_(-1/num_emb, 1/num_emb))

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')
       
        # distance = z**2 + e**2 - 2 e * z
        sq = torch.sum(self.code_book**2, dim=-1)[None, :] + torch.sum(x**2, dim=-1)[:, None]  # [b, k, c]
        dist = sq - 2*torch.einsum('bc, kc -> bk', x, self.code_book)

        # get nearest embedding
        ids = torch.argmin(dist, dim=-1)
        ids = rearrange(ids, '(b h w) -> b h w', b=b, h=h, w=w)
        emb = self.code_book[ids]
        emb = emb.permute(0, 3, 1, 2)

        return ids, emb
