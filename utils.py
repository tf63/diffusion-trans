import os
import glob

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def make_gif(out_dir, input_dir, img_name, wildcard='???.png', duration=50, delay=1000, reverse=False):
    imgs_name = glob.glob(os.path.join(input_dir, wildcard))
    imgs_name = sorted(imgs_name)

    imgs = list()
    for i in range(len(imgs_name)):
        imgs.append(Image.open(imgs_name[i]))

    duration_list = [duration] * len(imgs_name)
    duration_list[-1] = delay
    if reverse:
        imgs[-1].save(os.path.join(out_dir, f'{img_name}.gif'), save_all=True, loop=0, duration=duration_list,
                      append_images=reversed(imgs[0:-1]),)
    else:
        imgs[0].save(os.path.join(out_dir, f'{img_name}.gif'), save_all=True, loop=0, duration=duration_list,
                     append_images=imgs[1:],)


def make_gif_from_tensor(x, out_dir, img_name, duration=50, delay=1000, reverse=False):
    """
        out_dir: path/to/out_dir/
        img_name: img.gif
        x: (n, 3, 64, 64) 0.0 ~ 1.0
    """
    imgs = list()
    to_pil = torchvision.transforms.ToPILImage()
    for xi in x:
        imgs.append(to_pil(xi))

    duration_list = [duration] * len(imgs)
    duration_list[-1] = delay

    if reverse:
        imgs[-1].save(os.path.join(out_dir, f'{img_name}.gif'), save_all=True, loop=0, duration=duration_list,
                      append_images=reversed(imgs[0:-1]),)
    else:
        imgs[0].save(os.path.join(out_dir, f'{img_name}.gif'), save_all=True, loop=0, duration=duration_list,
                     append_images=imgs[1:],)


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.savefig('a.png')


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 3
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = '/home/tf63/project/nerf/diffusion/data/cifar10_64/cifar10-64'
    args.device = "cuda"
    args.lr = 3e-4

    dataloader = get_data(args)

    for image, label in dataloader:
        plt.imshow((image[0].permute(1, 2, 0) * 255).long())
        print(image[0].max(), image[0].min())
        plt.savefig('a.png')
        # save_images(image[0], 'a.png')
        break
