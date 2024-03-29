import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from loss import LPIPS


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
        dim=8,
        dim_mults=(2, 4, 8, 16, 32),
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
        dim=8,
        dim_mults=(2, 4, 8, 16, 32),
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


class Discriminator(nn.Module):
    """ PatchGAN Discriminator
    
    Refs:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    """
    def __init__(self, in_channels=3, dim=8, dim_mults=(2, 4, 8, 16, 32)):

        super().__init__()
        layers = [nn.Conv2d(in_channels, dim, 1), nn.LeakyReLU(0.2, True)]
        
        dims = [dim, *(m*dim for m in dim_mults)]
        n_resolutions = len(dim_mults)

        for i in range(n_resolutions):
            in_ch, out_ch = dims[i], dims[i + 1]
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True),
            ]
            in_ch = out_ch

        layers += [
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, 1, 1, 1), 
            nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class ResBlock(nn.Module):
#     def __init__(self, ch):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(ch, ch, 3, 1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(ch, ch, 1, 1),
#         )

#     def forward(self, x):
#         return x + self.layers(x)


# class Encoder(nn.Module):
#     """ Encoder: maps data 'x' to continous latent vectors 'ze'.
#     """
#     def __init__(self, in_ch=3, res_layers=2, hidden_ch=256, ):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_ch, hidden_ch // 2, 4, 2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_ch // 2, hidden_ch, 4, 2, padding=1),
#             nn.ReLU(),
#             *(ResBlock(hidden_ch) for _ in range(res_layers)),
#         )

#     def forward(self, x):
#         return self.layers(x)


# class Decoder(nn.Module):
#     """ Encoder: maps nearest embedding vectors 'zq' to data/output 'x'.
#     """
#     def __init__(self, in_ch=3, res_layers=2, hidden_ch=256, ):
#         super().__init__()
#         self.layers = nn.Sequential(
#             *(ResBlock(hidden_ch) for _ in range(res_layers)),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(hidden_ch, hidden_ch, 3, 1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_ch, in_ch, 1, 1),
#         )

#     def forward(self, x):
#         return self.layers(x)


# class Discriminator(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         nc, ndf = 3, 64
#         self.layers = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         y = self.layers(x)
#         return y.view(-1)


class VectorQuantizer(nn.Module):
    """ VQ: converts continous latents 'ze' to discrete latents 'z' then maps 'z' to nearest embedding vectors 'zq'. 
    """
    def __init__(self, num_emb, dimension, beta=0.25):
        super().__init__()
        self.code_book = nn.Parameter(torch.FloatTensor(num_emb, dimension).uniform_(-1/num_emb, 1/num_emb))
        self.beta = beta

    def forward(self, x, calculate_loss=True):
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

        emb_loss = None
        if calculate_loss:
            dict_loss = F.mse_loss(x.detach(), emb)
            commitment_loss = self.beta*F.mse_loss(x, emb.detach())
            emb_loss = dict_loss + commitment_loss

        # straight-through gradient hack - https://discuss.pytorch.org/t/relu-with-leaky-derivative/32818/2  
        emb = x + (emb - x).detach()
        
        return ids, emb, emb_loss

     
class VQGAN:
    def __init__(
        self, 
        in_ch=3, downsampling_factor=5, hidden_ch=256, num_emb=8*8,
        perceptual_loss=True,
        beta=0.25, lr=2e-4, device='cuda',
        ckpt=None, inference=False,
    ):
        self.in_ch = in_ch
        self.downsampling_factor = downsampling_factor
        self.hidden_ch = hidden_ch
        self.num_emb = num_emb
        self.perceptual_loss = perceptual_loss
        self.beta = beta
        self.lr = lr
        self.device = device
        if ckpt is None:
            dim = hidden_ch//(2**downsampling_factor)
            dim_mults = tuple(2**i for i in range(1, downsampling_factor + 1))
            self.enc = Encoder(in_ch, dim, dim_mults)
            self.vq  = VectorQuantizer(num_emb, hidden_ch, beta)
            self.dec = Decoder(in_ch, dim, dim_mults)
            self.disc = Discriminator(in_ch, dim, dim_mults).apply(weights_init)
            self.to(device)
            self.ae_params = list(self.enc.parameters()) + list(self.vq.parameters()) + list(self.dec.parameters())
            self.opt_ae = torch.optim.Adam(self.ae_params, lr=lr)
            self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=lr)
        else:
            self.load_ckpt(ckpt, device, inference)
        
        self.recon_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()
        if perceptual_loss:
            self.recon_loss = LPIPS().eval()
            self.recon_loss.to(device)

    # def __init__(
    #     self, 
    #     in_ch=3, res_layers=2, hidden_ch=256, num_emb=8*8*10, 
    #     perceptual_loss=True,
    #     beta=0.25, lr=2e-4, device='cuda',
    #     ckpt=None, inference=False,
    # ):
    #     super().__init__()
    #     self.in_ch = in_ch
    #     self.res_layers = res_layers
    #     self.hidden_ch = hidden_ch
    #     self.num_emb = num_emb
    #     self.beta = beta
    #     self.lr = lr
    #     self.device = device
    #     self.perceptual_loss = perceptual_loss
        
    #     if ckpt is None:
    #         self.enc = Encoder(in_ch, res_layers, hidden_ch)
    #         self.vq  = VectorQuantizer(num_emb, hidden_ch, beta)
    #         self.dec = Decoder(in_ch, res_layers, hidden_ch)
    #         self.disc = Discriminator().apply(weights_init)
    #         self.to(device)
    #         self.ae_params = list(self.enc.parameters()) + list(self.vq.parameters()) + list(self.dec.parameters())
    #         self.opt_ae = torch.optim.Adam(self.ae_params, lr=lr)
    #         self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=lr)
    #     else:
    #         self.load_ckpt(ckpt, device, inference)

    #     self.recon_loss = nn.MSELoss()
    #     self.adv_loss = nn.BCELoss()

    @property
    def n_ae_params(self):
        return sum(p.numel() for p in self.ae_params)

    @property
    def n_disc_params(self):
        return sum(p.numel() for p in self.disc.parameters())

    @property
    def n_params(self):
        return self.n_ae_params + self.n_disc_params

    def to(self, *args, **kwargs):
        self.enc.to(*args, **kwargs)
        self.vq.to(*args, **kwargs)
        self.dec.to(*args, **kwargs)
        self.disc.to(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.enc.train()
        self.vq.train()
        self.dec.train()
        self.disc.train()

    def eval(self, *args, **kwargs):
        self.enc.eval()
        self.vq.eval()
        self.dec.eval()
        self.disc.eval()
    
    def get_last_layer(self):
        return self.dec.final_conv[-1].weight

    # https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
    def calculate_adaptive_weight(self, rec_loss, gan_loss, last_layer):
        nll_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, x, calculate_loss=True, backward=True):

        ze = self.enc(x)
        z, zq, emb_loss = self.vq(ze, calculate_loss)
        x_ = self.dec(zq)
        lbl = self.disc(x_)

        recon_loss, gan_loss, loss, disc_loss = None, None, None, None
        if calculate_loss:
            real_lbl = torch.ones_like(lbl)
            fake_lbl = torch.zeros_like(lbl)
            
            recon_loss = self.recon_loss(x, x_)
            if self.perceptual_loss: recon_loss = recon_loss.mean()
            gan_loss = self.adv_loss(lbl, real_lbl)
            weight = self.calculate_adaptive_weight(recon_loss, gan_loss, self.get_last_layer()) if backward else 0.
            loss = recon_loss + emb_loss + weight*gan_loss
            # "inputs" arg is passed to stop grad accumulation on discriminator params
            # https://discuss.pytorch.org/t/how-to-implement-gradient-accumulation-for-gan/112751
            if backward: loss.backward(inputs=self.ae_params)

            real_loss = self.adv_loss(self.disc(x), real_lbl)
            fake_loss = self.adv_loss(self.disc(x_.detach()), fake_lbl)
            disc_loss = (real_loss + fake_loss)/2
            if backward: disc_loss.backward()
            
        return (ze, z, zq, x_, lbl), (recon_loss, emb_loss, gan_loss, loss), disc_loss
    
    def zero_grad(self, *args, **kwargs):
        self.opt_ae.zero_grad(*args, **kwargs)
        self.opt_disc.zero_grad(*args, **kwargs)

    def optimize(self, gradient_clip=None, new_lr=None, *args, **kwargs):
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.ae_params, gradient_clip)
            nn.utils.clip_grad_norm_(self.disc.parameters(), gradient_clip)

        if new_lr is not None:
            for param_group in self.opt_ae.param_groups:
                param_group['lr'] = new_lr

            for param_group in self.opt_disc.param_groups:
                param_group['lr'] = new_lr

        self.opt_ae.step()
        self.opt_disc.step()
        self.zero_grad(*args, **kwargs)

    # def get_config(self):
    #     cfg = {
    #         'in_ch':self.in_ch, 'res_layers':self.res_layers, 'hidden_ch':self.hidden_ch, 'num_emb':self.num_emb, 
    #         'beta':self. beta, 'lr': self.lr,
    #     }
    #     return cfg

    def get_config(self):
        cfg = {
            'in_ch': self.in_ch, 'downsampling_factor': self.downsampling_factor, 
            'hidden_ch': self.hidden_ch, 'num_emb': self.num_emb,
            'beta':self. beta, 'lr': self.lr,
        }
        return cfg

    def get_ckpt(self):
        ckpt = {
            'net':{
                'config': self.get_config(),
                'encoder': self.enc.state_dict(),
                'vector_quantizer': self.vq.state_dict(),
                'decoder': self.dec.state_dict(),
                'discriminator': self.disc.state_dict(),
            },
            'optimizer': {
                'AE': self.opt_ae.state_dict(),
                'discriminator': self.opt_disc.state_dict(),
            }
        }
        return ckpt

    def load_ckpt(self, ckpt, device='cpu', inference=False):
        if not isinstance(ckpt, dict): ckpt = torch.load(ckpt, map_location='cpu')
        config, enc, vq, dec, disc = ckpt['net'].values()
        for k, v in config.items():
            setattr(self, k, v)
        dim = self.hidden_ch//(2**self.downsampling_factor)
        dim_mults = tuple(2**i for i in range(1, self.downsampling_factor + 1))
        self.enc = Encoder(self.in_ch, dim, dim_mults)
        self.vq  = VectorQuantizer(self.num_emb, self.hidden_ch, self.beta)
        self.dec = Decoder(self.in_ch, dim, dim_mults)
        self.disc = Discriminator(self.in_ch, dim, dim_mults).apply(weights_init)
        self.enc.load_state_dict(enc)
        self.vq.load_state_dict(vq)
        self.dec.load_state_dict(dec)
        self.disc.load_state_dict(disc)
        self.to(device)
        self.ae_params = list(self.enc.parameters()) + list(self.vq.parameters()) + list(self.dec.parameters())
        print(f'Model loaded successfully ...')
        
        if (not inference) and ('optimizer' in ckpt):
            self.opt_ae = torch.optim.Adam(self.ae_params, lr=self.lr)
            self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
            self.opt_ae.load_state_dict(ckpt['optimizer']['AE'])
            self.opt_disc.load_state_dict(ckpt['optimizer']['discriminator'])
            print(f'Optimizer loaded successfully ...')

    def save_ckpt(self, filename):
        ckpt = self.get_ckpt()
        torch.save(ckpt, filename)