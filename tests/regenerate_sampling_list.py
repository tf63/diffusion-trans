import os
import glob

import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from utils import make_gif, make_gif_from_tensor, save_images
from modules import UNet_conditional
from ddpm_conditional import Diffusion


if __name__ == '__main__':

    # load model
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load('models/trained_cifar10/conditional_ckpt.pt')
    model.load_state_dict(ckpt)
    model.eval()

    # load image
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

    label = 1
    # label_cond = torch.tensor([1]).to(device)
    label_cond = torch.tensor([label]).to(device)
    i_img = 0
    i_run = 3
    # for cfg_scale in range(3, 10, 2):
    # for renoise_scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # imgs_path = sorted(glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*'))
    # img = Image.open(imgs_path[i_img])

    # noise step
    noise_steps = 1000
    save_step = 5

    # ----------------------------------------------------------------
    t_start = 500  # 再生成の開始ステップ
    # n_guid_noise = 20
    cfg_scale = 3
    n = 4
    begin = 0
    exp_name = f'sampling_regenerate_{label}_cfg{cfg_scale}'  # 何をしたいか

    t_starts = [100, 200, 300, 400, 500]
    run_name = f'label{label}_{label_cond[0]}_cfg{cfg_scale}_t{t_start}_list_run{i_run}'  # パラメータ
    out_dir = f'{os.path.abspath(".")}/results/test/{exp_name}/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/img', exist_ok=True)
    # ----------------------------------------------------------------
    # transform
    diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
    # to_tensor = torchvision.transforms.ToTensor()
    # img = to_tensor(img)[None]
    # img = img.to(device)
    # img = (img - 0.5) * 2

    with torch.no_grad():
        x_forward = torch.zeros((n * (len(t_starts) + 1), 3, 64, 64)).to(device)
        x_reverse = torch.zeros((n * (len(t_starts) + 1), 3, 64, 64)).to(device)

        # sampling ----------------------------------------------------------------
        x_sampling = torch.zeros((noise_steps + 1, n, 3, 64, 64)).to(device)

        x = torch.randn((n, 3, 64, 64)).to(device)
        for i in tqdm(reversed(range(0, noise_steps)), position=0):
            ti = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, ti, label_cond)
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

            x_sampling[i + 1] = x

            if (i % save_step == 0):
                x_plot = x
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}.png')

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')

        x_out = (x.clamp(-1, 1) + 1) / 2
        x_out = (x_out * 255).type(torch.uint8)
        save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out_sampling.png')

        for i in range(n):
            x_forward[i * (len(t_starts) + 1)] = x[i]
            x_reverse[i * (len(t_starts) + 1)] = x[i]

        # forward ----------------------------------------------------------------
        x_t = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
        # x_t[0] = img[0].tile(n, 1, 1, 1)
        x_t[0] = x
        # x = x_t[0]
        # Forward
        for i in range(t_start):
            if i > begin:
                epsilon_t = torch.randn_like(x)
                x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t
            else:
                # guid forward
                ti = (torch.tensor([i])).long().to(device)
                predicted_noise = model(x, ti, label_cond)
                # if cfg_scale > 0:
                #     uncond_predicted_noise = model(x, ti, None)
                #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = diffusion.alpha[ti][:, None, None, None]
                alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
                beta = diffusion.beta[ti][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # renoise ----------------------------------------------------------------
                # predicted_noise += renoise_scale * torch.rand_like(predicted_noise)
                # predicted_noise -= renoise_scale * torch.rand_like(predicted_noise)
                # ------------------------------------------------------------------------
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) - torch.sqrt(beta) * noise

            x_t[i + 1] = x

        x_plot = (x.clamp(-1, 1) + 1) / 2
        x_plot = (x_plot * 255).type(torch.uint8)
        save_images(x_plot, f'{out_dir}/forward_t{t_start}.png')

        for i in range(n):
            for j, t in enumerate(t_starts):
                x_forward[i * (len(t_starts) + 1) + j + 1] = x_t[t][i]

        # Reverse
        for j, t in enumerate(t_starts):
            x = x_t[t]
            for i in tqdm(reversed(range(t))):
                ti = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, ti, label_cond)
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

            x_plot = (x.clamp(-1, 1) + 1) / 2
            x_plot = (x_plot * 255).type(torch.uint8)
            save_images(x_plot, f'{out_dir}/out_t{t}.png')

            for i in range(n):
                x_reverse[i * (len(t_starts) + 1) + j + 1] = x[i]

        x_plot = (x_reverse.clamp(-1, 1) + 1) / 2
        x_plot = (x_plot * 255).type(torch.uint8)
        save_images(x_plot, f'test/img/run{i_run}_cfg{cfg_scale}_sampling_reverse_n{n}.pdf', nrow=6)
        x_plot = (x_forward.clamp(-1, 1) + 1) / 2
        x_plot = (x_plot * 255).type(torch.uint8)
        save_images(x_plot, f'test/img/run{i_run}_cfg{cfg_scale}_sampling_forward_n{n}.pdf', nrow=6)
