# Denoising Diffusion Probabilistic Models - DDPM
This repository contains implementation of Diffusion based Generative Model from scratch in PyTorch.

## Forward Diffusion
* pseudo code
## Reverse Diffusion
* pseudo code


# Getting Started

```shell
git clone https://github.com/girishdhegde/diffusion.git
cd ddpm
```

## Requirements
* python >= 3.9.13
* pytorch >= 1.13.1

## Installation
```
    pip install -r requirements.txt
```

# Usage
## Project Structure
```bash
ddpm
  ├── data.py - dataset, dataloader, collate_fn
  ├── utils.py - save/load ckpt, log prediction, sample from model
  ├── model.py - model
  ├── train.py - training loop
  └── demo.ipynb - inference and visualization
```
## Model Import
```python
from Model import UNet, DenoiseDiffusion
```

## Training and Inference
* train.py in each algorith subdirectory has training code.
* edit the **ALL_CAPITAL** parameters section at the starting of train.py as required. 
* demo.ipynb in each algorithm subdirectory has inference code.

# Codes Implemented

# Results

## License - MIT

# References