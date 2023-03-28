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
    label_cond = torch.tensor([0]).to(device)
    # label_cond = torch.tensor([label]).to(device)
    i_img = 0
    for _ in range(1):
        imgs_path = sorted(glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*'))
        img = Image.open(imgs_path[i_img])
        # img_path = '/home/tfukuda/project/nerf/diffusion_lab/results/test/sampling_1_cfg3/label1_1_cfg3__run6_out.png'
        # img_path = '/home/tf63/project/nerf/diffusion/data/cifar10_64/cifar10-64/test/class1/img104.png'
        # img = Image.open(img_path)
        # noise step
        noise_steps = 1000
        save_step = 5

        # ----------------------------------------------------------------
        t_start = 500  # 再生成の開始ステップ
        # n_guid_noise = 20
        cfg_scale = 0
        n = 1
        begin = 0
        exp_name = f'regenerate_recov{i_img}'  # 何をしたいか

        # l_loop = [10, 100, 150, 200, 300, 400, 500]
        # n_loop = [10, 1, 1, 1, 1, 1, 1]

        i_run = 5
        l_loop = [400, 100] * 1
        n_loop = [1, 5] * 1
        # cfg_scales = [cfg_scale] * len(l_loop)
        cfg_scales = [cfg_scale, cfg_scale] * 1
        # run_name = f'label{label}_{label_cond[0]}_cfg{cfg_scale}_lloop{l_loop}_nloop{n_loop}_run{i_run}'  # パラメータ
        run_name = f'label{label}_{label_cond[0]}_cfg{cfg_scale}_run{i_run}'  # パラメータ

        out_dir = f'{os.path.abspath(".")}/results/test/{exp_name}/{run_name}'

        os.makedirs(out_dir, exist_ok=True)
        # ----------------------------------------------------------------
        # transform
        diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
        to_tensor = torchvision.transforms.ToTensor()
        img = to_tensor(img)[None]
        img = img.to(device)
        img = (img - 0.5) * 2

        with torch.no_grad():

            # forward ----------------------------------------------------------------
            x = img[0].tile(n, 1, 1, 1)

            x_plot = (x.clamp(-1, 1) + 1) / 2
            x_plot = (x_plot * 255).type(torch.uint8)
            save_images(x_plot, f'{out_dir}/gt.png')

            for li in range(len(l_loop)):
                for loop in range(n_loop[li]):
                    print(f'{loop}: regenerate, 0 to {l_loop[li]}')
                    # Forward
                    for i in range(l_loop[li]):
                        if i > begin:
                            epsilon_t = torch.randn_like(x)
                            x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t
                        else:
                            # guid forward
                            ti = (torch.tensor([i])).long().to(device)
                            predicted_noise = model(x, ti, label_cond)
                            if cfg_scales[li] > 0:
                                uncond_predicted_noise = model(x, ti, None)
                                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
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

                    # Reverse
                    for i in tqdm(reversed(range(l_loop[li]))):
                        ti = (torch.ones(n) * i).long().to(device)
                        predicted_noise = model(x, ti, label_cond)
                        if cfg_scales[li] > 0:
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
                    save_images(x_plot, f'{out_dir}/{li}_{l_loop[li]}_itr{loop}.png')

            # # reverse ----------------------------------------------------------------
            # x = x_t[t_start - 1].to(device)
            # x_reverse = torch.zeros((t_start, n, 3, 64, 64))
            # noise_reverse = torch.zeros((t_start, n, 3, 64, 64))
            # for i in tqdm(reversed(range(1, t_start)), position=0):
            #     ti = (torch.tensor([i])).long().to(device)
            #     predicted_noise = model(x, ti, label_cond)
            #     if cfg_scale > 0:
            #         uncond_predicted_noise = model(x, ti, None)
            #         predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            #     alpha = diffusion.alpha[ti][:, None, None, None]
            #     alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            #     beta = diffusion.beta[ti][:, None, None, None]
            #     # noise = noise_forward_param[i]
            #     if i > 1:
            #         noise = torch.randn_like(x)
            #     else:
            #         noise = torch.zeros_like(x)
            #     x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            #     x_reverse[i] = x[0]
            #     noise_reverse[i] = predicted_noise[0]
            #     if i % save_step == 0 or i == 1:
            #         x_plot = torch.cat([x, x_t[i]], dim=0)
            #         # x_plot /= 2.
            #         x_plot = (x_plot.clamp(-1, 1) + 1) / 2
            #         x_plot = (x_plot * 255).type(torch.uint8)
            #         save_images(x_plot, f'{out_dir}/img/{(i):03d}_{t_start}_reverse.png', nrow=n)

            # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_reverse{t_start}', wildcard=f'???_{t_start}_reverse.png', reverse=True)
            # # x = torch.cat([x, img], dim=0)
            # x = (x.clamp(-1, 1) + 1) / 2
            # x = (x * 255).type(torch.uint8)
            # save_images(x, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out.png', nrow=n)
