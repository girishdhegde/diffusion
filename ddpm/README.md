# [Denoising Diffusion Probabilistic Models - DDPM](https://arxiv.org/pdf/2006.11239.pdf)
This repository contains implementation of Diffusion based Generative Model from scratch in PyTorch.
* **Autoregressive** network which **refines** **noise iteratively**
    ```python
    noise = gausian()
    for t in steps:
        less_noisy = model(noise, t)
        noise = less_noisy
    ```

## Forward Diffusion
* process of **addition of noise**
* $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$
* $x_t = \sqrt{1 - \beta_t} * x_{t-1} + \sqrt{\beta_t} * \mathcal{N}(0, I)$
* $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$; $\alpha_t := 1 - \beta_t$ and $\bar{\alpha}_t$ = product($\alpha_t$)
* $x_t =  \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} * \mathcal{N}(0, I)$ 
* `xt = root_alpha_cum_prods*xstart + root_one_minus_apha_cum_prods*noise`
* **Training**

    1. Repeat
    2. Let $x_0$ be a sample from $q(x_0)$
    3. Let $t$ be a sample from $\mathrm{Uniform}(\{1, \dotsc, T\})$
    4. Let $\epsilon$ be a sample from $\mathcal{N}(0, I)$
    5. $x_t =  \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} * \epsilon$ 
    6. Take gradient descent step on 
            $\nabla_\theta$ MSE{ $\epsilon$, $\epsilon_\theta$(xt, t)}
            where $\epsilon_\theta$   is a function approximator intended to predict $\epsilon$ from $x_t$. 
    7. Until converged
    ```python
    while not converged:
        xstart = sample(dataset)
        t = uniform(0, T)
        noise = gausian(0, I)
        noisy_data = diffuse(xstart, t, mean, variance)
        eps = model(noisy_data, t)
        loss = MSE(eps, noise)
        gradients = loss.backward()
        optimize(model, gradients)
    ```


## Reverse Diffusion
* process of **removal of noise**
* Using **Bayes** theorem, one can calculate the **posterior** $q(x_{t-1} | x_t, x_0)$ in terms of $\tilde{\beta}_t$ and $\tilde{\mu}_t(x_t, x_0)$ which are defined as follows:

$$
\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \quad \quad 
$$

$$
\tilde{\mu}_t(x_t, x_0) := \sqrt{\frac{\bar{\alpha}_{t-1}\beta_t}{1 - \bar{\alpha}_t}} x_0 + \sqrt{\frac{\alpha_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}} x_t \quad \quad 
$$

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t I) \quad \quad
$$
* **Sampling**

    1. $xT ∼ N(0, I)$
    2. for t = T, ..., 1 do
    3. $z ∼ N(0, I)$ if t > 1, else $z = 0$
    4. $x_{t-1} = \sqrt{\frac{1}{\alpha_t}}\left(x_t - \sqrt{\frac{1-\alpha_t}{1-\bar{\alpha}t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$
            `xt = recip_root_alphas[t]*(xt - betas_by_cum_prods[t]*eps) + sigmas[t]*z`

    5. end for
    6. return x0

# Getting Started

```shell
git clone https://github.com/girishdhegde/diffusion.git
pip install -r requirements.txt
cd ddpm
```

## Requirements
* python >= 3.9.13
* pytorch >= 1.13.1


# Usage
## Project Structure
```bash
ddpm
  ├── data.py - dataset, dataloader, collate_fn
  ├── utils.py - save/load ckpt, log prediction
  ├── model.py - UNet with attention, forward diffusion, reverse diffusion
  ├── train.py - training loop
  └── demo.ipynb - inference and visualization
```
## Model Import
```python
from Model import UNet, DenoiseDiffusion

net = UNet(
    in_channels,  # input image channels.
    out_channels,  # output channels.
    dim,  # hidden layer channels.
    dim_mults,  # hidden channel layerwise multipliers.
    attns,  # apply attention to corresponding layers if True.
    n_blocks,  # no. of res blocks per stage.
    groups,  # groupnorm num_groups.
)

denoiser = DenoiseDiffusion(net, timesteps)

xt, noise = denoiser.forward_sample(xstart, t)
samples = denoiser.reverse_sample(
    shape=(n_samples, channels, imgsize, imgsize), 
)
```
refer respective module **docstring** for **more info**.

## Training and Inference
* `python train.py`
* edit the **ALL_CAPITAL** parameters section at the starting of train.py as required. 
* `demo.ipynb` jupyter notebook has inference and visualization code.

# Codes Implemented
* UNet with Attention & Time Embedding
* Cosine Schedule
* Forward Diffusion
* Reverse Diffusion
* Diffusion Visualization

# Results
* Samples from **22M** model trained on **800** pokemons for **100K** steps with batch size of **16**
<img src="https://github.com/girishdhegde/diffusion/blob/master/ddpm/assets/samples.PNG"/>

* Rerverse Diffusion visualization
<img src="https://github.com/girishdhegde/diffusion/blob/master/ddpm/assets/denoising.PNG"/>

## License - MIT

# References
* https://arxiv.org/pdf/2006.11239.pdf
* https://arxiv.org/pdf/2102.09672.pdf
* https://arxiv.org/pdf/2301.11093.pdf
* https://github.com/hojonathanho/diffusion/issues/5
* https://github.com/lucidrains/denoising-diffusion-pytorch
* https://huggingface.co/blog/annotated-diffusion
* https://chat.openai.com