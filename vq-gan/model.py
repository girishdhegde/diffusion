import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from loss import LPIPS


    
class VQGAN:
    def __init__(
        self, 
        in_ch=3, downsampling_factor=5, hidden_ch=256, num_emb=8*8*10,
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
            
    def forward(self, x):

        ze = self.enc(x)
        z, zq, emb_loss = self.vq(ze)
        x_ = self.dec(zq)
        lbl = self.disc(x_)

        real_lbl = torch.ones_like(lbl)
        fake_lbl = torch.zeros_like(lbl)
        
        recon_loss = self.recon_loss(x, x_)
        if self.perceptual_loss: recon_loss = recon_loss.mean()
        gan_loss = self.adv_loss(lbl, real_lbl).mean()
        loss = recon_loss + emb_loss + gan_loss
        # "inputs" arg is passed to stop grad accumulation on discriminator params
        # https://discuss.pytorch.org/t/how-to-implement-gradient-accumulation-for-gan/112751
        loss.backward(inputs=self.ae_params)

        real_loss = self.adv_loss(self.disc(x), real_lbl)
        fake_loss = self.adv_loss(self.disc(x_.detach()), fake_lbl)
        disc_loss = (real_loss + fake_loss)/2
        disc_loss.backward()

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