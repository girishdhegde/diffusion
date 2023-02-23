import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


__author__ = "__Girish_Hegde__"


def extract(a, t, x_shape):
    """
    Refs:
        https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


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
        https://github.com/lucidrains/denoising-diffusion-pytorch
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

    Refs:
        https://github.com/lucidrains/denoising-diffusion-pytorch
        https://huggingface.co/blog/annotated-diffusion
    """
    def __init__(
        self,
        in_channels, out_channels,
        time_channels=None,
        groups=8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if time_channels is not None:
            self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels*2))
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
            scale, shift = t.chunk(chunks=2, dim=1)
            y = y*(scale + 1) + shift
        y = self.conv2(self.act1(y))
        return y + self.shortcut(x)


class PreNorm(nn.Module):
    """ https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Attention(nn.Module):
    """ Multi Headed Scaled Dot-Product Attention.

    Args:
        in_channels (int): input feature channels.
        emb_dim (int): dimension.
        heads (int): number of heads. (dq = dk = dv = d = emb_dim/h).
    """
    def __init__(self, in_channels, emb_dim, heads=1):
        super().__init__()
        self.heads = heads
        self.scale = (emb_dim//heads)**-0.5
        self.to_qkv = nn.Linear(in_channels, 3*emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, in_channels, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        QKV = self.to_qkv(x)  # [b h w 3*emb_dim]
        Q, K, V = QKV.chunk(3, dim=-1)  # [b h w emb_dim]
        # [b, h, w, emb_dim] -> [b*heads, h*w, d_head]
        Q, K, V = (rearrange(T, 'b h w (n d) -> (b n) (h w) d', n=self.heads, h=h) for T in (Q, K, V))
        attn = torch.bmm(Q, K.permute(0, 2, 1))*self.scale  # [b*heads, h*w, h*w]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)  # [bs*heads, h*w, d_head]
        out = rearrange(out, '(b n) (h w) d -> b h w (n d)', n=self.heads, h=h)  # [b, h, w, emb_dim]
        out = rearrange(self.proj(out), 'b h w c -> b c h w')

        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, groups, attn=False, downsample=True):
        super().__init__()
        self.resize = downsample
        self.conv = ResBlock(in_channels, out_channels, time_channels, groups)
        self.attn = PreNorm(out_channels, Attention(out_channels, 128, heads=4)) if attn else nn.Identity()
        self.ds = Downsample(out_channels, out_channels) if downsample else nn.Identity()

    def forward(self, x, time_emb=None):
        x = self.conv(x, time_emb)
        x = self.attn(x)
        ds = self.ds(x)
        return x, ds


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, groups, attn=False, upsample=True):
        super().__init__()
        self.resize = upsample
        self.conv = ResBlock(in_channels, out_channels, time_channels, groups)
        self.attn = PreNorm(out_channels, Attention(out_channels, 128, heads=4)) if attn else nn.Identity()
        self.us = Upsample(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x, time_emb=None):
        x = self.conv(x, time_emb)
        x = self.attn(x)
        x = self.us(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, groups):
        super().__init__()
        self.pre_conv = ResBlock(in_channels, out_channels, time_channels, groups)
        self.attn = PreNorm(out_channels, Attention(out_channels, 128, heads=4))
        self.post_conv = ResBlock(out_channels, out_channels, time_channels, groups)

    def forward(self, x, time_emb=None):
        x = self.pre_conv(x, time_emb)
        x = self.attn(x)
        x = self.post_conv(x, time_emb)
        return x


class UNet(nn.Module):
    """ UNet with Attention and Time embedding.

    Args:
        in_channels (int): input image channels.
        out_channels (int): output channels.
        dim (int): hidden layer channels.
        dim_mults (tuple[int]): hidden channel layerwise multipliers.
        attns (tuple[bool]): apply attention to corresponding layers if True.
        n_blocks (int): no. of res blocks per stage.
        groups (int): gropnorm num_groups.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=None,
        dim=16,
        dim_mults=(1, 2, 4, 8),
        attns=(False, False, True, True),
        n_blocks=1,
        groups=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dim = dim
        self.dim_mults = dim_mults
        self.attns = attns
        self.n_blocks = n_blocks
        self.groups = groups
        time_dim = dim*4

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, dim, 1)
        self.final_res = ResBlock(dim*2, dim, time_dim, groups)
        self.final_conv = nn.Conv2d(dim, self.out_channels, 1)

        dims = [dim, *(m*dim for m in dim_mults)]
        n_resolutions = len(dim_mults)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(n_resolutions):
            in_ch, out_ch, attn = dims[i], dims[i + 1], attns[i]
            for j in range(n_blocks):
                ds = (i < (n_resolutions - 1)) and (j == (n_blocks - 1))
                self.downs.append(
                    DownBlock(in_ch, out_ch, time_dim, groups, attn, ds)
                )
                in_ch = out_ch

        self.mid = MidBlock(dims[-1], dims[-1], time_dim, groups)

        for i in range(n_resolutions - 1, -1, -1):
            in_ch, out_ch, attn = dims[i + 1]*2, dims[i], attns[i]
            for j in range(n_blocks):
                us = (i > 0) and (j == (n_blocks - 1))
                self.ups.append(
                    UpBlock(in_ch, out_ch, time_dim, groups, attn, us)
                )
                in_ch = out_ch

    def forward(self, x, time):
        """
        Args:
            x (torch.tensor): [b, in_channels, h, w] - batch input images.
            time (torch.tensor): [b, ] - batch time stamps.

        Returns:
            torch.tensor: [b, out_channels, h, w] - output tensor.
        """
        t = self.time_emb(time)
        x = self.init_conv(x)
        skips = [x]

        for layer in self.downs:
            h, x = layer(x, t)
            if layer.resize:
                skips.append(h)
        skips.append(x)

        x = self.mid(x, t)

        concat = True
        for layer in self.ups:
            if concat:
                x = torch.cat((x, skips.pop()), dim=1)
                concat = False
            if layer.resize: concat = True
            x = layer(x, t)

        x = torch.cat((x, skips.pop()), dim=1)
        x = self.final_res(x, t)
        x = self.final_conv(x)
        return x


def linear_schedule(start=0.0001, end=0.02, timesteps=100):
    return np.linspace(start, end, timesteps)


def cosine_schedule(timesteps=100):
    steps = timesteps + 1
    t = 0.5*math.pi*((torch.linspace(0, timesteps, steps)/timesteps) + 0.008)/(1.008)
    ft = torch.cos(t)**2
    alphas_ = ft/ft[0]
    betas = 1 - (alphas_[1:]/alphas_[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DenoiseDiffusion:
    def __init__(self, model, timesteps=100, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        self.betas = cosine_schedule(timesteps)
        self.sigmas = torch.sqrt(self.betas)
        self.alphas = 1. - self.betas
        self.recip_root_alphas = 1/torch.sqrt(self.alphas)
        self.alpha_cum_prods = torch.cumprod(self.alphas, dim=0)
        self.root_alpha_cum_prods = torch.sqrt(self.alpha_cum_prods)
        self.root_one_minus_apha_cum_prods = torch.sqrt(1 - self.alpha_cum_prods)
        self.betas_by_cum_prods = self.betas/self.root_one_minus_apha_cum_prods

    @torch.no_grad()
    def forward_sample(self, xstart, t, noise=None):
        """ Forward Diffuion - Addition of Noise.

        Args:
            xstart (torch.Tensor): [b, ...] - input data.
            t (torch.LongTensor/List[int]): [b, ] - timesteps.
            noise (torch.Tensor): [b, ...] noise (noise.shape == xstart.shape).

        Returns:
            tuple
                torch.Tensor: xt - [b, ...] - noisy data.
                torch.Tensor: noise - [b, ...] - noise.
        """
        noise = torch.randn_like(xstart) if noise is None else noise
        root_alpha_cum_prods = self.root_alpha_cum_prods[t].clone()
        root_one_minus_apha_cum_prods = self.root_one_minus_apha_cum_prods[t].clone()
        while root_alpha_cum_prods.ndim < xstart.ndim:
            root_alpha_cum_prods = root_alpha_cum_prods[..., None]
            root_one_minus_apha_cum_prods = root_one_minus_apha_cum_prods[..., None]
        xt = root_alpha_cum_prods*xstart + root_one_minus_apha_cum_prods*noise
        return xt, noise

    @torch.no_grad()
    def reverse_sample(self, xt, time=None, return_timesteps=False):
        """ Reverse Diffusion - Removal of Noise.

        Args:
            xt (torch.Tensor): [b, ...] - noisy data.
            time (torch.LongTensor): [b, ] - timesteps.
            return_timesteps (bool): return denoised data for each timesteps.

        Returns:
            torch.Tensor: [b, ...] if not return_timesteps else [b, t, ...] - denoised data.
        """
        time = time or self.timesteps
        recip_root_alphas = self.recip_root_alphas.clone()
        betas_by_cum_prods = self.betas_by_cum_prods.clone()
        sigmas = self.sigmas.clone()
        while recip_root_alphas.ndim < (xt.ndim + 1):
            recip_root_alphas = recip_root_alphas[..., None]
            betas_by_cum_prods = betas_by_cum_prods[..., None]
            sigmas = sigmas[..., None]

        out = [xt]
        for i, t in enumerate(range(time - 1, 0, -1)):
            z = torch.rand_like(xt)
            eps = self.model(xt, t)
            xt = recip_root_alphas[t]*(xt - betas_by_cum_prods[t]*eps) + sigmas[t]*z
            if return_timesteps: out.append(xt)
        xt = recip_root_alphas[0]*(xt - betas_by_cum_prods[0]*self.model(xt, 0))
        
        if return_timesteps: 
            out.append(xt)
            return torch.cat(out, dim=1)

        return xt
