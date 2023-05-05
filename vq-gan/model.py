import torch
import torch.nn as nn
import torch.nn.functional as F
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

    
class VQVAE(nn.Module):
    def __init__(
        self, 
        in_ch=3, res_layers=2, hidden_ch=256, num_emb=8*8*10, 
        beta=0.25, lr=2e-4, device='cuda',
        ckpt=None, inference=False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.res_layers = res_layers
        self.hidden_ch = hidden_ch
        self.num_emb = num_emb
        self.beta = beta
        self.lr = lr
        self.device = device

        if ckpt is None:
            self.enc = Encoder(in_ch, res_layers, hidden_ch)
            self.vq  = VectorQuantizer(num_emb, hidden_ch, beta)
            self.dec = Decoder(in_ch, res_layers, hidden_ch)
            self.to(device)
            self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.load_ckpt(ckpt, inference)

    def forward(self, x):
        ze = self.enc(x)
        z, zq, emb_loss = self.vq(ze)
        x_ = self.dec(zq)

        recon_loss = F.mse_loss(x, x_)
        loss = recon_loss + emb_loss
        return (ze, z, zq, x_), loss

    def get_config(self):
        cfg = {
            'in_ch':self.in_ch, 'res_layers':self.res_layers, 'hidden_ch':self.hidden_ch, 'num_emb':self.num_emb, 
            'beta':self.beta, 'lr':self.lr, 
        }
        return cfg

    def get_ckpt(self):
        ckpt = {
            'net':{
                'config': self.get_config(),
                'encoder': self.enc.state_dict(),
                'vector_quantizer': self.vq.state_dict(),
                'decoder': self.dec.state_dict(),
            },
            'optimizer': self.opt.state_dict(),
        }
        return ckpt

    def load_ckpt(self, ckpt, inference=False):
        if not isinstance(ckpt, dict): ckpt = torch.load(ckpt, map_location='cpu')
        config, enc, vq, dec = ckpt['net'].values()
        for k, v in config.items():
            setattr(self, k, v)
        self.enc = Encoder(self.in_ch, self.res_layers, self.hidden_ch)
        self.vq  = VectorQuantizer(self.num_emb, self.hidden_ch, self.beta)
        self.dec = Decoder(self.in_ch, self.res_layers, self.hidden_ch)
        self.enc.load_state_dict(enc)
        self.vq.load_state_dict(vq)
        self.dec.load_state_dict(dec)
        self.to(self.device)
        print(f'Model loaded successfully ...')

        if (not inference) and ('optimizer' in ckpt):
            self.opt = torch.optim.Adam(self.parameters(), lr=1e-1)
            self.opt.load_state_dict(ckpt['optimizer'])
            print(f'Optimizer loaded successfully ...')

    def save_ckpt(self, filename):
        ckpt = self.get_ckpt()
        torch.save(ckpt, filename)
    
    def zero_grad(self, *args, **kwargs):
        self.opt.zero_grad(*args, **kwargs)

    def optimize(self, gradient_clip=None, new_lr=None, *args, **kwargs):
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.opt.parameters(), gradient_clip)

        if new_lr is not None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = new_lr

        self.opt.step()
        self.zero_grad(*args, **kwargs)