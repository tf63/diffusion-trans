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
    i_img = 0
    imgs_path = sorted(glob.glob(f'data/cifar10_64/cifar10-64/test/class{label}/*'))
    img = Image.open(imgs_path[i_img])
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)[None]
    img = (img - 0.5) * 2
    img = torch.cat([img] * 12, dim=0)
    print(img.shape)
    x_plot = img
    x_plot = (x_plot.clamp(-1, 1) + 1) / 2
    x_plot = (x_plot * 255).type(torch.uint8)
    save_images(x_plot, path=f'test/img/out12.pdf', nrow=6)
    # make_gif(out_dir=out_dir, input_dir=f'{out_dir}/img', img_name=f'{os.path.basename(out_dir)}_sampling', wildcard='???.png')
    # save_images(x_out, path=f'{os.path.dirname(out_dir)}/{os.path.basename(out_dir)}_out.pdf')
