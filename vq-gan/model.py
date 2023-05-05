import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


class ResBlock(nn.Module):
    """ Residual Block: conv2(scale_shit(conv(x), linear(time_emb))) + x

    Refs:
        https://github.com/lucidrains/denoising-diffusion-pytorch
        https://huggingface.co/blog/annotated-diffusion
    """
    def __init__(
        self,
        in_channels, out_channels,
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

    def forward(self, x):
        """
        Args:
            x (torch.tenosr): [b, c, h, w] - input features.

        Returns:
            torch.tensor: [b, c_, h, w] - output features.
        """
        y = self.norm1(self.conv1(x))
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
    def __init__(self, in_channels, out_channels, groups, downsample=True):
        super().__init__()
        self.resize = downsample
        self.conv = ResBlock(in_channels, out_channels, groups)
        self.ds = Downsample(out_channels, out_channels) if downsample else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.ds(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.res_init = ResBlock(channels, channels, groups)
        self.attn = PreNorm(channels, Attention(channels, channels, heads=8))
        self.res_final = ResBlock(channels, channels, groups)

    def forward(self, x):
        x = self.res_final(self.attn(self.res_init(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, upsample=True):
        super().__init__()
        self.resize = upsample
        self.conv = ResBlock(in_channels, out_channels, groups)
        self.us = Upsample(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.us(x)
        return x


class Encoder(nn.Module):
    """ Encoder with Attention.

    Args:
        in_channels (int): input image channels.
        dim (int): hidden layer channels.
        dim_mults (tuple[int]): hidden channel layerwise multipliers.
        n_blocks (int): no. of res blocks per stage.
        groups (int): groupnorm num_groups.
    """
    def __init__(
        self,
        in_channels=3,
        dim=16,
        dim_mults=(1, 2, 4, 8, 16),
        n_blocks=1,
        groups=4,
    ):
        super().__init__()
        assert dim%groups == 0, "dim must be divisible by groups"
        self.in_channels = in_channels
        self.out_channels = int(dim*dim_mults[-1])
        self.dim = dim
        self.dim_mults = dim_mults
        self.n_blocks = n_blocks
        self.groups = groups

        self.init_conv = nn.Conv2d(in_channels, dim, 1)
        self.attn_block = AttnBlock(self.out_channels, groups)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, self.out_channels), 
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 1)
        )

        dims = [dim, *(m*dim for m in dim_mults)]
        n_resolutions = len(dim_mults)
        self.downs = nn.ModuleList()

        for i in range(n_resolutions):
            in_ch, out_ch = dims[i], dims[i + 1]
            for j in range(n_blocks):
                ds = (j == (n_blocks - 1))
                self.downs.append(
                    DownBlock(in_ch, out_ch, groups, ds)
                )
                in_ch = out_ch

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [b, in_channels, h, w] - batch input images.

        Returns:
            torch.tensor: [b, out_channels, h_, w_] - output tensor.
        """
        x = self.init_conv(x)

        for layer in self.downs:
            x = layer(x)

        x = self.attn_block(x)
        x = self.final_conv(x)

        return x


class Decoder(nn.Module):
    """ Decoder with Attention.

    Args:
        out_channels (int): output image channels.
        dim (int): hidden layer channels.
        dim_mults (tuple[int]): hidden channel layerwise multipliers.
        n_blocks (int): no. of res blocks per stage.
        groups (int): groupnorm num_groups.
    """
    def __init__(
        self,
        out_channels=3,
        dim=16,
        dim_mults=(1, 2, 4, 8, 16),
        n_blocks=1,
        groups=4,
    ):
        super().__init__()
        assert dim%groups == 0, "dim must be divisible by groups"
        self.out_channels = out_channels
        self.in_channels = int(dim*dim_mults[-1])
        self.dim = dim
        self.dim_mults = dim_mults
        self.n_blocks = n_blocks
        self.groups = groups

        self.init_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)
        self.attn_block = AttnBlock(self.in_channels, groups)

        dims = [dim, *(m*dim for m in dim_mults)]
        n_resolutions = len(dim_mults)
        self.ups = nn.ModuleList()

        for i in range(n_resolutions - 1, -1, -1):
            in_ch, out_ch = dims[i + 1], dims[i]
            for j in range(n_blocks):
                us = (j == (n_blocks - 1))
                self.ups.append(
                    UpBlock(in_ch, out_ch, groups, us)
                )
                in_ch = out_ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, out_ch), 
            nn.SiLU(),
            nn.Conv2d(out_ch, self.out_channels, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [b, in_channels, h, w] - batch input images.

        Returns:
            torch.tensor: [b, out_channels, h_, w_] - output tensor.
        """
        x = self.init_conv(x)
        x = self.attn_block(x)

        for layer in self.ups:
            x = layer(x)

        x = self.final_conv(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
                https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=5):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False

        kw = 4
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        in_ch = 1
        out_ch = ndf
        for n in range(1, n_layers):  # gradually increase the number of filters
            in_ch = out_ch
            nf_mult = min(2 ** n, 8)
            out_ch = ndf * nf_mult
            out_ch = min(out_ch, 256)
            sequence += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kw, stride=2, padding=1, bias=use_bias),
                norm_layer(out_ch),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.Conv2d(out_ch, 1, kernel_size=1, stride=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class VectorQuantizer(nn.Module):
    """ VQ: converts continous latents 'ze' to discrete latents 'z' then maps 'z' to nearest embedding vectors 'zq'. 
    """
    def __init__(self, num_emb, dimension, beta=0.25):
        super().__init__()
        self.code_book = nn.Parameter(torch.FloatTensor(num_emb, dimension).uniform_(-1/num_emb, 1/num_emb))
        self.beta = beta

    def forward(self, x):
        b, c, h, w = x.shape
        x_ = rearrange(x, 'b c h w -> (b h w) c')
       
        # distance = z**2 + e**2 - 2 e * z
        sq = torch.sum(self.code_book**2, dim=-1)[None, :] + torch.sum(x_**2, dim=-1)[:, None]  # [b, k, c]
        dist = sq - 2*torch.einsum('bc, kc -> bk', x_, self.code_book)

        # get nearest embedding
        ids = torch.argmin(dist, dim=-1)
        ids = rearrange(ids, '(b h w) -> b h w', b=b, h=h, w=w)
        emb = self.code_book[ids]
        emb = emb.permute(0, 3, 1, 2)

        dict_loss = F.mse_loss(x.detach(), emb)
        commitment_loss = self.beta*F.mse_loss(x, emb.detach())
        emb_loss = dict_loss + commitment_loss

        # straight-through gradient hack - https://discuss.pytorch.org/t/relu-with-leaky-derivative/32818/2  
        emb = x + (emb - x).detach()
        
        return ids, emb, emb_loss

    
# class VQVAE(nn.Module):
#     def __init__(
#         self, 
#         in_ch=3, res_layers=2, hidden_ch=256, num_emb=8*8*10, 
#         beta=0.25, lr=2e-4, device='cuda',
#         ckpt=None, inference=False,
#     ):
#         super().__init__()
#         self.in_ch = in_ch
#         self.res_layers = res_layers
#         self.hidden_ch = hidden_ch
#         self.num_emb = num_emb
#         self.beta = beta
#         self.lr = lr
#         self.device = device

#         if ckpt is None:
#             self.enc = Encoder(in_ch, res_layers, hidden_ch)
#             self.vq  = VectorQuantizer(num_emb, hidden_ch, beta)
#             self.dec = Decoder(in_ch, res_layers, hidden_ch)
#             self.to(device)
#             self.opt = torch.optim.Adam(self.parameters(), lr=lr)
#         else:
#             self.load_ckpt(ckpt, inference)

#     def forward(self, x):
#         ze = self.enc(x)
#         z, zq, emb_loss = self.vq(ze)
#         x_ = self.dec(zq)

#         recon_loss = F.mse_loss(x, x_)
#         loss = recon_loss + emb_loss
#         return (ze, z, zq, x_), loss

#     def get_config(self):
#         cfg = {
#             'in_ch':self.in_ch, 'res_layers':self.res_layers, 'hidden_ch':self.hidden_ch, 'num_emb':self.num_emb, 
#             'beta':self.beta, 'lr':self.lr, 
#         }
#         return cfg

#     def get_ckpt(self):
#         ckpt = {
#             'net':{
#                 'config': self.get_config(),
#                 'encoder': self.enc.state_dict(),
#                 'vector_quantizer': self.vq.state_dict(),
#                 'decoder': self.dec.state_dict(),
#             },
#             'optimizer': self.opt.state_dict(),
#         }
#         return ckpt

#     def load_ckpt(self, ckpt, inference=False):
#         if not isinstance(ckpt, dict): ckpt = torch.load(ckpt, map_location='cpu')
#         config, enc, vq, dec = ckpt['net'].values()
#         for k, v in config.items():
#             setattr(self, k, v)
#         self.enc = Encoder(self.in_ch, self.res_layers, self.hidden_ch)
#         self.vq  = VectorQuantizer(self.num_emb, self.hidden_ch, self.beta)
#         self.dec = Decoder(self.in_ch, self.res_layers, self.hidden_ch)
#         self.enc.load_state_dict(enc)
#         self.vq.load_state_dict(vq)
#         self.dec.load_state_dict(dec)
#         self.to(self.device)
#         print(f'Model loaded successfully ...')

#         if (not inference) and ('optimizer' in ckpt):
#             self.opt = torch.optim.Adam(self.parameters(), lr=1e-1)
#             self.opt.load_state_dict(ckpt['optimizer'])
#             print(f'Optimizer loaded successfully ...')

#     def save_ckpt(self, filename):
#         ckpt = self.get_ckpt()
#         torch.save(ckpt, filename)
    
#     def zero_grad(self, *args, **kwargs):
#         self.opt.zero_grad(*args, **kwargs)

#     def optimize(self, gradient_clip=None, new_lr=None, *args, **kwargs):
#         if gradient_clip is not None:
#             nn.utils.clip_grad_norm_(self.opt.parameters(), gradient_clip)

#         if new_lr is not None:
#             for param_group in self.opt.param_groups:
#                 param_group['lr'] = new_lr

#         self.opt.step()
#         self.zero_grad(*args, **kwargs)