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
    # run name
    # ================================================================
    i_run = 0  # run number

    # seed
    torch.manual_seed(i_run)

    # load image directly
    # airplane to car
    label = torch.tensor([0]).to(device)
    label_cond = torch.tensor([1]).to(device)
    # img_name = 'img90'
    # img_name = 'img2185'
    # img_name = 'img2330'
    img_name = 'img9699'

    # car to airplane
    # label = torch.tensor([1]).to(device)
    # label_cond = torch.tensor([0]).to(device)
    # img_name = 'img104'
    # img_name = 'img330'
    # img_name = 'img659'
    # img_name = 'img7964'

    img_path = f'data/cifar10_64/cifar10-64/test/class{label[0]}/{img_name}.png'

    # load image from index
    # i_img = 0
    # imgs_path = glob.glob(f'data/cifar10_64/cifar10-64/test/class{label[0]}/*')
    # img_path = imgs_path[i_img]
    # img_name = os.path.basename(os.path.splitext(img_path)[0])

    img_mask_path = f'masks/{img_name}_mask.png'
    run_name = f'{img_name}_run{i_run}'
    # ================================================================
    # exp name
    # ================================================================
    exp_name = 'regenerate_mask'

    t_cfg = True
    cfg_scale = 5

    if t_cfg:
        exp_name += f'_t-cfg{cfg_scale}'
    else:
        exp_name += f'_cfg{cfg_scale}'
    exp_name += f'_{label[0]}to{label_cond[0]}'
    # exp_name += '_debug'
    # ================================================================
    # settings
    # ================================================================
    noise_steps = 1000
    save_step = 5
    save_x_t = False
    t_starts = [300]
    # t_starts = [100, 200, 300, 400]
    # t_starts = [5 * i for i in range(30)]
    # t_starts = [0, 100, 200, 300, 400, 500]
    n = len(t_starts)

    exp_dir = f'{os.path.abspath(".")}/results/test/{exp_name}'
    out_dir = f'{exp_dir}/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    if save_x_t:
        os.makedirs(f'{out_dir}/img', exist_ok=True)

    # load image
    img = Image.open(img_path)
    mask = Image.open(img_mask_path)

    # transform
    diffusion = Diffusion(noise_steps=noise_steps, img_size=64, device=device)
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)[None]
    img = img.to(device)
    img = (img - 0.5) * 2

    mask = to_tensor(mask)[None]
    mask = mask.to(device)
    mask = (mask - 0.5) * 2
    mask[mask < 0.8] = -1
    mask[mask > 0.8] = 1

    with torch.no_grad():

        # ================================================================
        # forward process
        # ================================================================
        print('Forward Process ----------------------------------------------------------------')

        x_t_forward = torch.zeros((noise_steps + 1, 1, 3, 64, 64)).to(device)
        x_t_forward[0] = img[0]
        x = x_t_forward[0]
        for i in range(noise_steps):
            epsilon_t = torch.randn_like(x)
            x = torch.sqrt(diffusion.alpha[i]) * x + torch.sqrt(1 - diffusion.alpha[i]) * epsilon_t

            if save_x_t and i % save_step == 0:
                x_plot = torch.cat([x, img[0]], dim=0)
                x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                x_plot = (x_plot * 255).type(torch.uint8)
                save_images(x_plot, f'{out_dir}/img/{i:03d}.png')

            x_t_forward[i + 1] = x

        # ================================================================
        # reverse process
        # ================================================================
        print('Reverse Process ----------------------------------------------------------------')
        for ni, t_start in enumerate(t_starts):
            x = x_t_forward[t_start]
            x_t_reverse = torch.zeros((t_start + 1, 1, 3, 64, 64)).to(device)
            x_t_reverse[t_start] = x
            for i in tqdm(reversed(range(1, t_start + 1))):
                ti = (torch.ones(1) * i).long().to(device)
                predicted_noise = model(x, ti, label_cond)
                if cfg_scale > 0:
                    if t_cfg:
                        uncond_predicted_noise = model(x, ti, label)
                    else:
                        uncond_predicted_noise = model(x, ti, None)

                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = diffusion.alpha[ti][:, None, None, None]
                alpha_hat = diffusion.alpha_hat[ti][:, None, None, None]
                beta = diffusion.beta[ti][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x_mask = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                # マスクされた領域のみを再生成
                x[mask > 0] = x_t_forward[i][mask > 0]
                x[mask < 0] = x_mask[mask < 0]

                x_t_reverse[i - 1] = x

                if save_x_t and i % save_step == 0:
                    x_plot = torch.cat([x, img[0]], dim=0)
                    x_plot = (x_plot.clamp(-1, 1) + 1) / 2
                    x_plot = (x_plot * 255).type(torch.uint8)
                    save_images(x_plot, f'{out_dir}/img/{(t_start - i) + t_start:03d}.png')

            x_gif = torch.cat([x_t_forward[:t_start - 1], x_t_reverse.flip(dims=[0])], dim=0)
            x_gif = (x_gif.clamp(-1, 1) + 1) / 2
            make_gif_from_tensor(x=x_gif[::save_step, 0, ...], out_dir=out_dir, img_name=f'{run_name}_t{t_start}', delay=1000)

            x_plot = (x.clamp(-1, 1) + 1) / 2
            x_plot = (x_plot * 255).type(torch.uint8)
            save_images(x_plot, f'{out_dir}/out_t{t_start}.png')

            x_out_list = torch.cat([img, mask, x], dim=0)
            x_plot = (x_out_list.clamp(-1, 1) + 1) / 2
            x_plot = (x_plot * 255).type(torch.uint8)
            save_images(x_plot, f'{exp_dir}/{run_name}_t{t_starts[ni]:03d}_list.png')
            save_images(x_plot, f'{out_dir}/{run_name}_t{t_starts[ni]:03d}_list.pdf')
