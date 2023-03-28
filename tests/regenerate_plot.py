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

    label = 1
    label_cond = torch.tensor([0]).to(device)
    # label_cond = torch.tensor([label]).to(device)
    i_img = 0
    i_run = 2
    # for cfg_scale in range(3, 10, 2):
    # for renoise_scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for i_img in range(1):
        imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*')
        img = Image.open(imgs_path[i_img])
        # noise step
        noise_steps = 1000
        save_step = 5

        # ----------------------------------------------------------------
        t_start = 1000  # 再生成の開始ステップ
        # n_guid_noise = 20
        cfg_scale = 3
        n = 1
        begin = 0
        exp_name = f'debug'  # 何をしたいか

        l_loop = 100
        n_loop = 1
        run_name = f'{i_img}_label{label}_{label_cond[0]}_cfg{cfg_scale}_lloop{l_loop}_nloop{n_loop}_run{i_run}'  # パラメータ
        out_dir = f'{os.path.abspath(".")}/results/test/{exp_name}/{run_name}'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(f'{out_dir}/img', exist_ok=True)
        # ----------------------------------------------------------------
        # transform
        diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
        to_tensor = torchvision.transforms.ToTensor()
        img = to_tensor(img)[None]
        img = img.to(device)
        img = (img - 0.5) * 2

        with torch.no_grad():

            # sampling ----------------------------------------------------------------
            x_sampling_t = torch.zeros((noise_steps, n, 3, 64, 64)).to(device)

            x = torch.randn((n, 3, 64, 64)).to(device)
            for i in tqdm(reversed(range(1, noise_steps))):
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

                x_sampling_t[i - 1] = x
                # print(f'{i} x: {x.max(), x.min()}')
                # print(f'{i} pred: {predicted_noise.max(), predicted_noise.min()}')
                if (i % save_step == 0 or i == 1):
                    x_plot = x
                    x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                    x_plot = (x_plot * 255).type(torch.uint8)
                    save_images(x_plot, f'{out_dir}/img/{(noise_steps - i):03d}.png')

            make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')

            x_out = (x.clamp(-1, 1) + 1) / 2
            x_out = (x_out * 255).type(torch.uint8)
            save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out_sampling.png')

            # forward ----------------------------------------------------------------
            x_t = torch.zeros((t_start + 1, n, 3, 64, 64)).to(device)
            noise_forward_t = torch.zeros((t_start + 1, 3, 64, 64)).to(device)
            # x_t[0] = img[0].tile(n, 1, 1, 1)
            # x = x_t[0]
            x_t[0] = x
            # Forward
            for i in range(t_start):
                if i > begin:
                    epsilon_t = torch.randn_like(x)
                    x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t
                    noise_forward_t[i + 1] = epsilon_t
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

            # noise_reverse_t = torch.zeros((t_start + 1, 3, 64, 64)).to(device)
            # # Reverse
            # # for t in [100, 200, 300, 400, 500]:
            # for t in [1000]:
            #     x = x_t[t]
            #     for i in tqdm(reversed(range(t))):
            #         ti = (torch.ones(n) * i).long().to(device)
            #         predicted_noise = model(x, ti, label_cond)
            #         if cfg_scale > 0:
            #             uncond_predicted_noise = model(x, ti, None)
            #             predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            #         alpha = diffusion.alpha[ti][:, None, None, None]
            #         alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
            #         beta = diffusion.beta[ti][:, None, None, None]
            #         if i > 1:
            #             noise = torch.randn_like(x)
            #         else:
            #             noise = torch.zeros_like(x)
            #         x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            #         noise_reverse_t[i] = predicted_noise
            #     x_plot = (x.clamp(-1, 1) + 1) / 2
            #     x_plot = (x_plot * 255).type(torch.uint8)
            #     save_images(x_plot, f'{out_dir}/out_t{t}.png')

            flatten = torch.nn.Flatten()
            # noise_forward = flatten(noise_forward_t) / 4
            # noise_reverse = flatten(noise_reverse_t) / 4
            # print(noise_forward.shape, noise_reverse.shape)
            # noise_forward_plot = torch.zeros(t_start)
            # noise_reverse_plot = torch.zeros(t_start)
            # for i in range(1, t_start):
            #     noise_forward_plot[i] = torch.dot(noise_forward[i], noise_forward[i - 1])
            #     noise_reverse_plot[i] = torch.dot(noise_reverse[i], noise_reverse[i - 1])

            x_forward = flatten(x_t) / 4
            x_sampling = flatten(x_sampling_t) / 4

            t = torch.arange(t_start)
            x_forward_plot = torch.zeros(t_start)
            x_reverse_plot = torch.zeros(t_start)
            for i in range(t_start):
                x_forward_plot[i] = torch.dot(x_forward[i], x_forward[0])
                x_reverse_plot[i] = torch.dot(x_sampling[i], x_sampling[0])

            # td = torch.arange(202, t_start)
            # print(t.shape, noise_forward_plot.shape)
            plt.figure(figsize=(6, 6))
            plt.rc('legend', fontsize=16)
            # plt.subplots_adjust(wspace=0.4)
            # plt.xlim(-50, 1050)
            # plt.ylim(-50, 1050)
            # plt.xticks([0, 200, 400, 600, 800, 1000])
            # plt.yticks([0, 200, 400, 600, 800, 1000])
            # plt.subplot(1, 2, 1)
            plt.xlabel('t', fontsize=16)
            plt.ylabel('類似度', fontsize=16)
            plt.plot(t, x_forward_plot[:], label='拡散過程', color='tab:blue')
            plt.plot(t, x_reverse_plot[:], label='逆過程', color='tab:orange')
            # plt.plot(t, noise_forward_plot[2:], label='拡散過程', color='tab:blue')
            # plt.plot(t, noise_reverse_plot[2:], label='逆過程', color='tab:orange')
            plt.legend(loc='upper right')
            # plt.subplot(1, 2, 2)
            # plt.xlabel('t', fontsize=16)
            # plt.ylabel('類似度', fontsize=16)
            # plt.plot(t, noise_reverse_plot[2:], label='逆過程', color='tab:orange')
            # plt.plot(td, noise_reverse_plot[202:], label='逆過程', color='tab:orange')
            # plt.legend(loc='lower right')
            # plt.savefig('similarity_forward_x.pdf')
            # plt.savefig('similarity_reverse_x.pdf')
            plt.savefig('similarity_forward_reverse_x3.pdf')
            # plt.savefig('similarity_forward_reverse.pdf')
            # plt.savefig('similarity_reverse_200.pdf')
            # plt.savefig('similarity_forward.pdf')
            # plt.show()

            # x_plot = torch.cat([noise_forward_t[100:106], noise_reverse_t[100:106]], dim=0)
            # x_plot /= 2
            # print(x_plot.max(), x_plot.min())
            # x_plot = (x_plot.clamp(-1, 1) + 1) / 2
            # x_plot = (x_plot * 255).type(torch.uint8)
            # save_images(x_plot, f'test/img/noise_forward_reverse.pdf', nrow=6)
            # print(x_plot.shape)
