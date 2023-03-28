import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, required=True,
                        help='run name')

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_size', type=int, default=64,
                        help='image size')
    parser.add_argument('--views', type=int, default=10,
                        help='number of views (each instance)')

    # train options
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--epoch_ini', type=int, default=-1,
                        help='initial epoch')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs')

    # sampling options
    parser.add_argument('--n_sample', type=int, default=10,
                        help='number of samples in sampling')
    parser.add_argument('--noise_step', type=int, default=1000,
                        help='noise step')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='classifier free guidance scale')

    # loss parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='lr')

    return parser.parse_args()
