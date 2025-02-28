import os
import glob
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from utils import make_gif, make_gif_from_tensor, save_images
from modules import UNet_conditional
from ddpm_conditional import Diffusion
# import japanize_matplotlib

if __name__ == '__main__':

    # load model
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load('models/trained_cifar10/conditional_ckpt.pt')
    model.load_state_dict(ckpt)
    model.eval()

    # cuDNN用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # label
    """
        label 0 -> 飛行機
        label 1 -> 車
        label 2 -> 鳥
        label 3 -> 猫
        label 4 -> 鹿
        label 5 -> 犬
        label 6 -> 蛙
        label 7 -> 馬
        label 8 -> 船
        label 9 -> トラック
    """
    # ================================================================
    # exp name
    # ================================================================

    exp_name = 'sampling'

    label = torch.tensor([1]).to(device)
    cfg_scale = 2
    exp_name += f'_cfg{cfg_scale}'
    exp_name += f'_{label[0]}'
    # exp_name += '_debug'
    # ================================================================
    # run name
    # ================================================================
    i_run = 2  # run number

    # seed
    torch.manual_seed(i_run)
    run_name = f'sample_cls{label[0]}_run{i_run}'

    print('================================================================')
    print(exp_name)
    print(run_name)
    print('================================================================')
    # ================================================================
    # settings
    # ================================================================
    noise_steps = 1000
    save_step = 10
    save_x_t = True
    n = 5  # n samples

    exp_dir = f'{os.path.abspath(".")}/results/test/{exp_name}'
    out_dir = f'{exp_dir}/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    if save_x_t:
        os.makedirs(f'{out_dir}/img', exist_ok=True)

    # transform
    diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
    with torch.no_grad():

        # ================================================================
        # reverse process
        # ================================================================
        print('Reverse Process ----------------------------------------------------------------')
        x = torch.randn((n, 3, 64, 64)).to(device)
        for i in tqdm(reversed(range(1, noise_steps)), position=0):
            ti = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, ti, label)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, ti, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

            alpha = diffusion.alpha[ti][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            beta = diffusion.beta[ti][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            if save_x_t and i % save_step == 0:
                x_plot = x
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{i:03d}.png')

        if save_x_t:
            make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{run_name}_step{noise_steps}', wildcard='???.png', delay=1000, reverse=True)

        x_plot = (x.clamp(-1, 1) + 1) / 2
        x_plot = (x_plot * 255).type(torch.uint8)
        save_images(x_plot, f'{exp_dir}/{run_name}_out.png')
        save_images(x_plot, f'{out_dir}/{run_name}_out.pdf')
