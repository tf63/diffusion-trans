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
    label_cond = torch.tensor([label]).to(device)
    # i_img = 10
    # imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*')
    # img = Image.open(imgs_path[i_img])

    # noise step
    noise_steps = 1000
    save_step = 5

    # ----------------------------------------------------------------
    t_start = 100  # 再生成の開始ステップ
    cfg_scale = 0
    # exp_name = f'regenerate_revnoise_for{i_img}'  # 何をしたいか
    # exp_name = 'sampling_regenerate_noiseshare_forward'  # 何をしたいか
    exp_name = 'sampling_regenerate_forward2'  # 何をしたいか
    i_run = 4
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

    # sampling ----------------------------------------------------------------
    x_sampling = torch.zeros((noise_steps, 3, 64, 64)).to(device)
    noise_sampling = torch.zeros(((noise_steps, 3, 64, 64))).to(device)
    noise_sampling_param = torch.zeros(((noise_steps, 3, 64, 64))).to(device)
    with torch.no_grad():
        x = torch.randn((1, 3, 64, 64)).to(device)
        for i in tqdm(reversed(range(1, noise_steps)), position=0):
            ti = (torch.tensor([i])).long().to(device)
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

            x_sampling[i] = x[0]
            noise_sampling[i] = predicted_noise[0]
            noise_sampling_param[i] = noise[0]

            if i % save_step == 0 or i == 1:
                x_plot = x / 2.
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}.png')
                noise_plot = predicted_noise / 2.
                noise_plot = (noise_plot.clamp(-1, 1) + 1) / 2
                noise_plot = (noise_plot * 255).type(torch.uint8)
                save_images(noise_plot, f'{out_dir}/img/{(noise_steps - i):03d}_noise.png')

        noise_diff = noise_sampling[1:-1] - noise_sampling[0:-2]
        noise_diff = noise_sampling - noise_sampling[0]
        noise_diff = torch.abs(noise_diff)
        for i in range(len(noise_diff)):
            # noise_diff[i] = noise_diff[i] / noise_diff[i].max()
            noise_diff[i] = noise_diff[i] / 6.
        noise_diff = noise_diff.clamp(0, 1)
        noise_diff = (noise_diff * 255).type(torch.uint8)
        save_images(noise_diff[1::save_step], path=f'{out_dir}/{os.path.basename(out_dir)}_noise_sampling_diff_list.png')
        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')
        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling_noise', wildcard='???_noise.png')
        make_gif_from_tensor(f'{out_dir}/{os.path.basename(out_dir)}_sampling_noise_diff.gif', noise_diff[::save_step])

        x_out = (x.clamp(-1, 1) + 1) / 2
        x_out = (x_out * 255).type(torch.uint8)
        save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out.png')

        # forward ----------------------------------------------------------------
        # img = x[0].tile((len(t), 1, 1, 1))
        # x_t, noise_forward = diffusion.noise_images(img, t)
        x_t = torch.zeros((t_start, 3, 64, 64)).to(device)
        noise_forward = torch.zeros((t_start, 3, 64, 64)).to(device)
        noise_forward_param = torch.zeros((t_start, 3, 64, 64)).to(device)
        x_t[0] = x

        for i in tqdm(range(1, t_start), position=0):
            x = x_t[i - 1][None].to(device)
            ti = (torch.tensor([i])).long().to(device)
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
            print(x[0].max(), x[0].min())
            x_t[i] = x[0]
            noise_forward[i] = predicted_noise[0]
            noise_forward_param[i] = noise[0]

            # print(x[0].max(), x[0].min())
            # print(f'alpha: {alpha}, alpha_hat: {alpha_hat}')
            # print(f'noise_max: {noise.max()}, noise_min: {noise.min()}')
            # print(f'predicted_noise_max: {predicted_noise.max()}, predicted_noise_min: {predicted_noise.min()}')
            if i % save_step == 0 or i == 1:
                x_plot = (x.clamp(-1, 1) + 1) / 2
                x_plot /= x_plot.max()
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}_forward.png')
                noise_plot = predicted_noise / 2.
                noise_plot = (noise_plot.clamp(-1, 1) + 1) / 2
                noise_plot = (noise_plot * 255).type(torch.uint8)
                save_images(noise_plot, f'{out_dir}/img/{(noise_steps - i):03d}_forward_noise.png')

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_forward', wildcard='???_forward.png', reverse=True)
        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_forward_noise', wildcard='???_forward_noise.png', reverse=True)

        noise_diff_sampling_forward = noise_forward - noise_sampling[:t_start]
        noise_diff_sampling_forward = torch.abs(noise_diff_sampling_forward)
        print(f'noise_forward: {noise_forward.max(), noise_forward.min()}')
        print(f'noise_sampling:  {noise_sampling.max(), noise_sampling.min()}')
        print(f'noise_diff_sampling: {noise_diff_sampling_forward.max(), noise_diff_sampling_forward.min()}')
        noise_diff_sampling_forward /= 6.
        noise_diff_sampling_forward = (noise_diff_sampling_forward * 255).type(torch.uint8)
        save_images(noise_diff_sampling_forward[1::save_step], path=f'{out_dir}/{os.path.basename(out_dir)}_noise_sampling_diff_list.png')
        make_gif_from_tensor(f'{out_dir}/{os.path.basename(out_dir)}_noise_diff_sampling_forward.gif', noise_diff_sampling_forward[::save_step])

        # reverse
        x = x_t[t_start - 1][None].to(device)
        x_reverse = torch.zeros((t_start, 3, 64, 64))
        noise_reverse = torch.zeros((t_start, 3, 64, 64))
        for i in tqdm(reversed(range(1, t_start)), position=0):
            ti = (torch.tensor([i])).long().to(device)
            predicted_noise = model(x, ti, label_cond)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, ti, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = diffusion.alpha[ti][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            beta = diffusion.beta[ti][:, None, None, None]
            # noise = noise_forward_param[i][None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) - torch.sqrt(beta) * noise

            x_reverse[i] = x[0]
            noise_reverse[i] = predicted_noise[0]
            if i % save_step == 0 or i == 1:
                x_plot = torch.stack([x[0], x_t[i]], dim=0)
                x_plot /= 2.
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}_reverse.png')
                noise_plot = torch.stack([predicted_noise[0], noise_forward[i]], dim=0)
                noise_plot /= 2.
                noise_plot = (noise_plot.clamp(-1, 1) + 1) / 2
                noise_plot = (noise_plot * 255).type(torch.uint8)
                save_images(noise_plot, f'{out_dir}/img/{(noise_steps - i):03d}_reverse_noise.png')

        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_reverse', wildcard='???_reverse.png')
        make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_reverse_noise', wildcard='???_reverse_noise.png')

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        save_images(x, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_reverse_out.png')
