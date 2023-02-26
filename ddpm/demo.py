from pathlib import Path

import numpy as np
import matplot.pyplot as plt
import torch

from model import DenoiseDiffusion, UNet


__author__ = "__Girish_Hegde__"


# =============================================================
# Parameters
# =============================================================
CKPT = Path('./data/runs/ckpt.pt')
DEVICE = torch.device('cpu')
CHANNELS = 3

# =============================================================
# Load Checkpoint
# =============================================================
ckpt = torch.load(CKPT, map_location='cpu')
net = UNet(**ckpt['net']['config'])
net.load_state_dict(ckpt['net']['state_dict'])
net.eval()
net.to(DEVICE)
timesteps = ckpt['kwargs']['timesteps']
imgsize = ckpt['kwargs']['img_size']
denoiser = DenoiseDiffusion(net, timesteps=timesteps)

# =============================================================
# Sample Images
# =============================================================
sample = denoiser.reverse_sample(xt=None, shape=(64, CHANNELS, imgsize, imgsize), return_timesteps=True)
sample = (sample*0.5) + 0.5
sample = (np.clip(sample.numpy(), 0, 1)*255).astype(np.uint8)
imgs = sample[-1]

# =============================================================
# Make Grid
# =============================================================
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.axis('off')

# =============================================================
# Save and Plot Samples
# =============================================================
plt.tight_layout()
fig.savefig(CKPT.parent/'samples.png')
plt.show()

# =============================================================
# Plot Denoising
# =============================================================
fig, axes = plt.subplots(nrows=1, ncols=timesteps + 1, figsize=(10, 10))
axes = axes.flatten()
idx = 0
for i, ax in enumerate(axes):
    ax.imshow(sample[i, idx])
    ax.axis('off')

# =============================================================
# Save and Plot Denoising Visualization
# =============================================================
plt.tight_layout()
fig.savefig(CKPT.parent/'denoising.png')
plt.show()

# =============================================================
# END
# =============================================================