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
    i_run = 0

    label = 1
    label_cond = torch.tensor([label]).to(device)
    i_img = 32
    imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*')
    img = Image.open(imgs_path[i_img])
    # noise step
    noise_steps = 1000
    save_step = 5

    # ----------------------------------------------------------------
    t_start = 500  # 再生成の開始ステップ
    cfg_scale = 0
    n = 5
    # exp_name = f'regenerate_revnoise_for{i_img}'  # 何をしたいか
    exp_name = 'sampling_guided_noforward_samplingnoiseshare_debug'  # 何をしたいか
    # exp_name = f'sampling_guided_forward_list_shared_+noise{i_img}'  # 何をしたいか

    run_name = f'label{label}-{label_cond[0]}_t{t_start}-{noise_steps}_cfg{cfg_scale}_run{i_run}'  # パラメータ
    out_dir = f'{os.path.abspath(".")}/results/test/{exp_name}/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/img', exist_ok=True)
    # ----------------------------------------------------------------
    # cfg_scale_forward = -0.3
    # plusloop = 100
    # run_name = f'label{label}-{label_cond[0]}_t{t_start}-{noise_steps}_cfg{cfg_scale}_{cfg_scale_forward}_{plusloop}++'  # パラメータ
    # ----------------------------------------------------------------

    # setting t
    t = torch.arange(0, t_start)
    t = t.to(device)

    # transform
    diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)[None]
    img = img.to(device)
    img = (img - 0.5) * 2

    with torch.no_grad():

        # sampling ----------------------------------------------------------------
        x_sampling = torch.zeros((noise_steps, n, 3, 64, 64)).to(device)
        noise_sampling = torch.zeros(((noise_steps, n, 3, 64, 64))).to(device)
        noise_sampling_param = torch.zeros(((noise_steps, n, 3, 64, 64))).to(device)

        x = torch.randn((n, 3, 64, 64)).to(device)
        for i in tqdm(reversed(range(1, noise_steps)), position=0):
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

            x_sampling[i] = x
            noise_sampling[i] = predicted_noise
            noise_sampling_param[i] = noise

            if i < t_start and (i % save_step == 0 or i == 1):
                # x_plot = x / 2.
                x_plot = x
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}.png')

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')

        x_out = (x.clamp(-1, 1) + 1) / 2
        x_out = (x_out * 255).type(torch.uint8)
        save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out_sampling.png')

        # forward ----------------------------------------------------------------
        x_t = torch.zeros((t_start, n, 3, 64, 64)).to(device)
        noise_forward = torch.zeros((t_start, n, 3, 64, 64)).to(device)
        noise_forward_param = torch.zeros((t_start, n, 3, 64, 64)).to(device)
        # x_t[0] = img[0].tile(n, 1, 1, 1)
        x_t[0] = x
        for i in tqdm(range(1, t_start), position=0):
            # ti = (torch.tensor([i])).long().to(device)
            # predicted_noise = model(x, ti, label_cond)
            # if cfg_scale > 0:
            #     uncond_predicted_noise = model(x, ti, None)
            #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            # alpha = diffusion.alpha[ti][:, None, None, None]
            # alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            # beta = diffusion.beta[ti][:, None, None, None]
            # noise = noise_sampling_param[i]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) - torch.sqrt(beta) * noise
            # # print(x[0].max(), x[0].min())
            # noise_forward[i] = predicted_noise
            # noise_forward_param[i] = noise

            # no forward ----------------------------------------------------------------
            epsilon_t = noise
            epsilon_t_minus_1 = noise_forward[i - 1]
            x = torch.sqrt(diffusion.alpha[i])[None, None] * (x - torch.sqrt(1 - diffusion.alpha_hat[i - 1])[None, None] * epsilon_t_minus_1)
            x += torch.sqrt(1 - diffusion.alpha_hat[i])[None, None] * epsilon_t
            noise_forward[i] = epsilon_t
            noise_forward_param[i] = torch.randn_like(x)
            # ----------------------------------------------------------------

            x_t[i] = x
            if i % save_step == 0 or i == 1:
                x_plot = (x_t[i].clamp(-1, 1) + 1) / 2
                # x_plot /= x_plot.max()
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}_forward.png', nrow=n)

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_forward', wildcard='???_forward.png', reverse=True)

        # reverse ----------------------------------------------------------------
        x = x_t[t_start - 1].to(device)
        x_reverse = torch.zeros((t_start, n, 3, 64, 64))
        noise_reverse = torch.zeros((t_start, n, 3, 64, 64))
        for i in tqdm(reversed(range(1, t_start)), position=0):
            ti = (torch.tensor([i])).long().to(device)
            predicted_noise = model(x, ti, label_cond)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, ti, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = diffusion.alpha[ti][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            beta = diffusion.beta[ti][:, None, None, None]
            noise = noise_forward_param[i]
            # if i > 1:
            #     noise = torch.randn_like(x)
            # else:
            #     noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            x_reverse[i] = x[0]
            noise_reverse[i] = predicted_noise[0]
            if i % save_step == 0 or i == 1:
                x_plot = torch.cat([x, x_t[i]], dim=0)
                # x_plot /= 2.
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}_reverse.png', nrow=n)

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_reverse', wildcard='???_reverse.png')
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        x = torch.cat([x, x_out], dim=0)
        save_images(x, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out.png', nrow=n)
