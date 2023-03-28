import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from utils import make_gif, make_gif_from_tensor, save_images
from modules import UNet_conditional
from ddpm_conditional import Diffusion
import japanize_matplotlib

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

    label = 0
    label_cond = torch.tensor([1]).to(device)
    # label_cond = torch.tensor([label]).to(device)
    i_img = 10
    i_run = 6
    # for cfg_scale in range(3, 10, 2):
    # for renoise_scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for _ in range(1):
        # imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*')
        img_name = 90
        img_path = f'data/cifar10_64/cifar10-64/test/class{label}/img{img_name}.png'
        img_mask_path = f'data/img{img_name}_mask.png'
        print(f'image: {img_path}')
        img = Image.open(img_path)
        mask = Image.open(img_mask_path)
        # noise step
        noise_steps = 1000
        save_step = 5

        # ----------------------------------------------------------------
        t_start = 300  # 再生成の開始ステップ
        # n_guid_noise = 20
        cfg_scale = 0
        n = 1
        begin = 0
        exp_name = f'regenerate_trans_cfg_debug{img_name}'  # 何をしたいか
        run_name = f'{i_img}_label{label}_{label_cond[0]}_cfg{cfg_scale}_t{t_start}_run{i_run}'  # パラメータ
        out_dir = f'{os.path.abspath(".")}/results/test/{exp_name}/{run_name}'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(f'{out_dir}/img', exist_ok=True)
        print(f'out_dir: {out_dir}')
        # ----------------------------------------------------------------
        # transform
        diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
        to_tensor = torchvision.transforms.ToTensor()
        img = to_tensor(img)[None]
        img = img.to(device)
        img = (img - 0.5) * 2
        mask = to_tensor(mask)[None]
        # mask = torch.zeros_like(img)
        mask = mask.to(device)
        mask = (mask - 0.5) * 2
        mask[mask < 0.8] = -1
        mask[mask > 0.8] = 1
        with torch.no_grad():

            # forward ----------------------------------------------------------------
            x_t = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
            noise_forward_t = torch.zeros((t_start + 1, 3, 64, 64)).to(device)
            x_t[0] = img[0].tile(n, 1, 1, 1)
            x = x_t[0]
            # Forward
            print('Forward Process ---------------------------------------------')
            for i in tqdm(range(t_start)):
                # if i % 1 == 0:
                if True:
                    epsilon_t = torch.randn_like(x)
                    x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t
                    noise_forward_t[i + 1] = epsilon_t
                    # x = x_t[i - 1][None].to(device)
                else:
                    ti = (torch.tensor([i - 1])).long().to(device)
                    epsilon_t = model(x, ti, label_cond)
                    predicted_noise = epsilon_t
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, ti, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = diffusion.alpha[ti][:, None, None, None]
                    alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
                    beta = diffusion.beta[ti][:, None, None, None]
                    epsilon_t = predicted_noise
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    # x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * predicted_noise
                    # print(predicted_noise.max(), predicted_noise.min())
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    # x = torch.sqrt(alpha) * x + (1 - alpha) / (torch.sqrt(1 - alpha_hat)) * predicted_noise + torch.sqrt(alpha * beta) * noise
                    # x = (torch.sqrt(alpha) * x - torch.sqrt(alpha * beta)) / predicted_noise + (1 - alpha) / (torch.sqrt(1 - alpha_hat))
                    print(f'{i} x: {x[0].max(), x[0].min()}')
                    print(f'{i} pred: {predicted_noise.max(), predicted_noise.min()}')
                    print(f'{i} noise: {noise.max(), noise.min()}')
                    # print(f'alpha: {1 / torch.sqrt(alpha)}')
                    # print(f'alpha, s: {(1 - alpha) / (torch.sqrt(1 - alpha_hat))}')
                    # print(f'beta: {torch.sqrt(beta)}')

                x_t[i + 1] = x
                noise_forward_t[i + 1] = epsilon_t
                if i % save_step == 0:
                    x_plot = torch.cat([x, x_t[0]], dim=0)
                    # x_plot = x
                    x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                    x_plot = (x_plot * 255).type(torch.uint8)
                    save_images(x_plot, f'{out_dir}/img/{i:03d}.png')
                    noise_plot = epsilon_t / 2
                    noise_plot = (noise_plot.clamp(-1, 1) + 1) / 2
                    noise_plot = (noise_plot * 255).type(torch.uint8)
                    save_images(noise_plot, f'{out_dir}/img/{i:03d}_noise_forward.png')
                    # x_plot = x_plot.cpu().numpy()
                    # bins = np.linspace(0, 255)
                    # plt.hist([x_plot[0][0], x_plot[0][1], x_plot[0][2]])
                    # plt.hist(x_plot[0][0])
                    # plt.savefig(f'{out_dir}/img/{i:03d}_hist_red.png')
                    # plt.cla()
                    # plt.hist(x_plot[0][1])
                    # plt.savefig(f'{out_dir}/img/{i:03d}_hist_green.png')
                    # plt.cla()
                    # plt.hist(x_plot[0][2])
                    # plt.savefig(f'{out_dir}/img/{i:03d}_hist_blue.png')
                    # plt.cla()
                # x_t[i + 1] = x

            # x[mask > 0] = 1
            noise_reverse_t = torch.zeros((t_start + 1, 3, 64, 64)).to(device)
            # Reverse
            for t in [t_start]:
                # x = x_t[t]
                print(f'Reverse Process {t_start} ------------------------------------------')
                for i in tqdm(reversed(range(1, t + 1))):
                    ti = (torch.ones(n) * i).long().to(device)
                    # predicted_noise = model(x, ti, label_cond)
                    predicted_noise = model(x, ti, None)
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

                    pred = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    # x = pred
                    x[mask > 0] = x_t[i][mask > 0]
                    x[mask < 0] = pred[mask < 0]

                    # x[mask > 0] = pred[mask > 0]
                    # x[mask < 0] = x_t[i][mask < 0]

                    # noise_reverse_t[i - 1] = predicted_noise
                    # print(x[0].max(), x[0].min())

                    if i % save_step == 0:
                        x_plot = torch.cat([x, x_t[0]], dim=0)
                        # x_plot = x
                        x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                        x_plot = (x_plot * 255).type(torch.uint8)
                        save_images(x_plot, f'{out_dir}/img/{(t - i) + t:03d}.png')
                        # noise_plot = torch.cat([noise_forward_t[i][None], predicted_noise[0][None]], dim=0)
                        # # noise_plot = predicted_noise
                        # noise_plot /= 2
                        # noise_plot = (noise_plot.clamp(-1, 1) + 1) / 2
                        # noise_plot = (noise_plot * 255).type(torch.uint8)
                        # save_images(noise_plot, f'{out_dir}/img/{i:03d}_noise.png')

                make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}', wildcard='???.png', delay=2000)
                # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_noise', wildcard='???_noise.png', delay=50)
                # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_hist_red', wildcard='???_hist_red.png', delay=100)
                # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_hist_green', wildcard='???_hist_green.png', delay=100)
                # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_hist_blue', wildcard='???_hist_blue.png', delay=100)
                x_plot = torch.cat([img, mask, x], dim=0)
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/out_t{t}.pdf')
                save_images(x_plot, f'{out_dir}/out_t{t}.png')
