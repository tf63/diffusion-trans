# %%
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging

# %%
import glob
from PIL import Image

def make_gif(path):
    imgs_name = glob.glob(os.path.join(path, '???.png'))
    imgs_name = sorted(imgs_name)
    imgs = list()
    for i in range(len(imgs_name)):
        imgs.append(Image.open(imgs_name[i]))
        imgs[i] = imgs[i].resize((imgs[i].width, imgs[i].height))
        duration = [50] * len(imgs_name)
        duration[-1] = 3000
        imgs[0].save(os.path.join(os.path.dirname(path), f'{os.path.basename(path)}.gif'), save_all=True, loop=0, duration=duration,
                        append_images=imgs[1:],)

# %%

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        print(self.alpha_hat.shape)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, gif_name, gif_div=10, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        os.makedirs(f'img/{gif_name}', exist_ok=True)
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                img = (x.clamp(-1, 1) + 1) / 2
                img = (img * 255).type(torch.uint8)
                if i % gif_div == 0 or i == 1:
                    save_images(img, f'img/{gif_name}/{(self.noise_steps - i):03d}.png')

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)


        return x

# %%
device = 'cuda'
model = UNet_conditional(num_classes=10).to(device)
ckpt = torch.load('models/trained_cifar10/conditional_ema_ckpt.pt')
model.load_state_dict(ckpt)

# %%
### save images
n = 5
cfg_scale = 3
noise_steps = 1000
diffusion = Diffusion(img_size=64, device=device, noise_steps=noise_steps)
n_class = 1
img_name = f'cifar10_class{n_class}_cfg{cfg_scale}_{n}sample_step{noise_steps}'
y = torch.Tensor([n_class] * n).long().to(device)
x = diffusion.sample(model, n, y, gif_name=img_name, gif_div=20, cfg_scale=cfg_scale)
make_gif(path=f'img/{img_name}')
save_images(x, f'img/{img_name}.png')


# %%
imgs_path_class0 = glob.glob('data/cifar10_64/cifar10-64/test/class0/*')
img = Image.open(imgs_path_class0[0])

noise_steps = 1000
# img.save('a.png')
to_tensor = torchvision.transforms.ToTensor()
img = to_tensor(img)
img = (img[None] * 255).type(torch.uint8)

# save_images(img[None], f'a_tensor.png')

# %%
t = torch.arange(20, noise_steps + 1, 20)
img = img[0].tile((50, 1, 1, 1))
print(img.shape)
print(t.shape)
print(t)

# %%
noise_steps = 1000
img = img.to(device)
# t
diffusion = Diffusion(img_size=64, device=device, noise_steps=noise_steps)
# x_t, noise = diffusion.noise_images(img, t)
# t = diffusion.sample_timesteps(50)
print(t)



