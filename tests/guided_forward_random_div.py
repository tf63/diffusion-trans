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
    i_img = 17
    i_run = 1
    for cfg_scale in range(3, 10, 2):
        # for renoise_scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*')
        img = Image.open(imgs_path[i_img])
        # noise step
        noise_steps = 1000
        save_step = 5

        # ----------------------------------------------------------------
        t_step = 1
        t_start = 500  # 再生成の開始ステップ
        # n_guid_noise = 20
        n_guid_noise = 1
        # cfg_scale = 3.0
        n = 1
        guid_reverse = True
        # i_run = 3
        # exp_name = f'regenerate_revnoise_for{i_img}'  # 何をしたいか
        exp_name = f'cfg_guided_forward{i_img}'  # 何をしたいか
        # exp_name = f'sampling_guided_forward_list_shared_+noise{i_img}'  # 何をしたいか

        # run_name = f'label{label}-{label_cond[0]}_t{t_start}-{noise_steps}_cfg{cfg_scale}_run{i_run}_t-step{t_step}_n-gnoise{n_guid_noise}'  # パラメータ
        if guid_reverse:
            run_guid = 'guid'
        else:
            run_guid = 'noguid'
        # renoise_scale = 0.1
        run_name = f'cfg{cfg_scale}_label{label}-{label_cond[0]}_n-gnoise{n_guid_noise}_t-step{t_step}_{run_guid}_run{i_run}'  # パラメータ
        # run_name = f'n-gnoise{n_guid_noise}_t-step{t_step}_{run_guid}{renoise_scale}_run{i_run}'  # パラメータ
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
            # x_sampling = torch.zeros((noise_steps, n, 3, 64, 64)).to(device)
            # noise_sampling = torch.zeros(((noise_steps, n, 3, 64, 64))).to(device)
            # noise_sampling_param = torch.zeros(((noise_steps, n, 3, 64, 64))).to(device)

            # x = torch.randn((n, 3, 64, 64)).to(device)
            # for i in tqdm(reversed(range(1, noise_steps)), position=0):
            #     ti = (torch.ones(n) * i).long().to(device)
            #     predicted_noise = model(x, ti, label_cond)
            #     if cfg_scale > 0:
            #         uncond_predicted_noise = model(x, ti, None)
            #         predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            #     alpha = diffusion.alpha[ti][:, None, None, None]
            #     alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            #     beta = diffusion.beta[ti][:, None, None, None]
            #     if i > 1:
            #         noise = torch.randn_like(x)
            #     else:
            #         noise = torch.zeros_like(x)
            #     x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            #     x_sampling[i] = x
            #     noise_sampling[i] = predicted_noise
            #     noise_sampling_param[i] = noise

            #     if i < t_start and (i % save_step == 0 or i == 1):
            #         # x_plot = x / 2.
            #         x_plot = x
            #         x_plot = (x_plot.clamp(-1, 1) + 1) / 2
            #         x_plot = (x_plot * 255).type(torch.uint8)
            #         save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}.png')

            # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')

            # x_out = (x.clamp(-1, 1) + 1) / 2
            # x_out = (x_out * 255).type(torch.uint8)
            # save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out_sampling.png')

            # forward ----------------------------------------------------------------
            x_t = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
            # noise_forward = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
            # noise_forward_param = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
            x_t[0] = img[0].tile(n, 1, 1, 1)
            # x_t[0] = x
            loss = torch.nn.MSELoss()

            for k in range(t_start // t_step):  # どの区間か
                print(f'Forward Process: {k * t_step} to {(k + 1) * t_step}')
                min_mse = 100000

                for _ in tqdm(range(n_guid_noise)):  # どのノイズか
                    x = x_t[k * t_step]
                    x_t_noise = torch.zeros_like(x_t)
                    # Forward Process
                    # まとめてノイズを付加することもできる
                    # 別々でやってるのはvalidation用
                    if not guid_reverse:
                        for i in range(k * t_step, (k + 1) * t_step):  # Forward Process
                            epsilon_t = torch.randn_like(x)

                            x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t

                            x_t_noise[i + 1] = x
                    else:
                        for i in range(k * t_step, (k + 1) * t_step):  # Guided Forward Process
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
                            # ----------------------------------------------------------------
                            # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) - torch.sqrt(beta) * noise

                            x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * predicted_noise
                            x_t_noise[i + 1] = x

                    # Reverse Process
                    for i in reversed(range(k * t_step, (k + 1) * t_step)):  # Reverse Process
                        ti = (torch.ones(n) * i).long().to(device)
                        predicted_noise = model(x, ti, label_cond)
                        if cfg_scale > 0:
                            uncond_predicted_noise = model(x, ti, None)
                            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                        alpha = diffusion.alpha[ti][:, None, None, None]
                        alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
                        beta = diffusion.beta[ti][:, None, None, None]
                        # noise = noise_forward_param[i]  # 今のところ意味ない
                        if i > 1:
                            noise = torch.randn_like(x)
                        else:
                            noise = torch.zeros_like(x)
                        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                    # 損失が小さかったら
                    mse = loss(x, x_t[k * t_step])
                    if min_mse > mse:
                        # x_tの更新
                        print(f'update x_t at {k} iteration, mse: {min_mse} to {mse}')
                        min_mse = mse
                        x_t[k * t_step + 1:(k + 1) * t_step + 1] = x_t_noise[k * t_step + 1:(k + 1) * t_step + 1]

            # ここでx_tを保存しとく ?
            for i in range(0, t_start):
                if i % save_step == 0:
                    x_plot = (x_t[i].clamp(-1, 1) + 1) / 2
                    # x_plot /= x_plot.max()
                    x_plot = (x_plot * 255).type(torch.uint8)
                    save_images(x_plot, f'{out_dir}/img/{(i):03d}_forward.png', nrow=n)

            make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_forward', wildcard='???_forward.png')

            # reverse ----------------------------------------------------------------
            for t_start in [100, 200, 300, 400, 500]:
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
                    # noise = noise_forward_param[i]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                    x_reverse[i] = x[0]
                    noise_reverse[i] = predicted_noise[0]
                    if i % save_step == 0 or i == 1:
                        x_plot = torch.cat([x, x_t[i]], dim=0)
                        # x_plot /= 2.
                        x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                        x_plot = (x_plot * 255).type(torch.uint8)
                        save_images(x_plot, f'{out_dir}/img/{(i):03d}_{t_start}_reverse.png', nrow=n)

                make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_reverse{t_start}', wildcard=f'???_{t_start}_reverse.png', reverse=True)
                x = torch.cat([x, img], dim=0)
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                save_images(x, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_{t_start}_out.png', nrow=n)
