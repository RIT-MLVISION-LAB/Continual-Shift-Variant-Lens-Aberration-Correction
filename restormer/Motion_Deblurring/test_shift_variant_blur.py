### Example code for testing Shift-Variant Blur Deblurring using Restormer
### PYTHONPATH=.. python test_shift_variant_blur.py \
###    --weights ../experiments/Shift_Variant_V1_Deblurring_Restormer/models/net_g_148000.pth \
###    --variant 1 --split val --save_images
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Shift-Variant Blur Deblurring using Restormer')
    parser.add_argument('--input_dir', default='./Datasets/', type=str,
                        help='Directory of validation/test images')
    parser.add_argument('--result_dir', default='./results/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--variant', type=int, default=1,
                        help='Variant number (1-6)')
    parser.add_argument('--split', default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate (default: val)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save restored images')

    args = parser.parse_args()

    # loading model configuration
    yaml_file = f'Options/Deblurring_ShiftVariant_V{args.variant}.yml'
    if not os.path.exists(yaml_file):
        yaml_file = 'Options/Deblurring_Restormer.yml'
        print(f"Warning: Variant config not found, using default: {yaml_file}")

    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(yaml_file, mode='r') as f:
        config = yaml.load(f, Loader=Loader)

    # building model
    network_config = config['network_g'].copy()
    network_config.pop('type', None)
    model = Restormer(**network_config)

    # loading weights
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded weights from: {args.weights}")

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    # setting up paths
    dataset_name = f'ShiftVariant_V{args.variant}_Full_Images'
    inp_dir = os.path.join(args.input_dir, args.split, dataset_name, 'input_crops')
    gt_dir = os.path.join(args.input_dir, args.split, dataset_name, 'target_crops')
    result_dir = os.path.join(args.result_dir, dataset_name)

    if args.save_images:
        os.makedirs(result_dir, exist_ok=True)

    # checking if GT exists for metric computation
    has_gt = os.path.exists(gt_dir)

    # getting input files
    files = natsorted(glob(os.path.join(inp_dir, '*.png')))

    if len(files) == 0:
        print(f"No images found in {inp_dir}")
        return

    print(f"Testing on {len(files)} images from {dataset_name} ({args.split})")

    factor = 8  # padding factor for Restormer

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # loading input image
            img = np.float32(load_img(file_)) / 255.
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img_tensor.unsqueeze(0).cuda()

            # padding to multiple of 8
            h, w = input_.shape[2], input_.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            # forward pass
            restored = model(input_)

            # removing padding
            restored = restored[:, :, :h, :w]

            # converting to numpy
            restored = torch.clamp(restored, 0, 1).cpu().detach()
            restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()

            # computes metrics if GT available
            if has_gt:
                filename = os.path.basename(file_)
                gt_path = os.path.join(gt_dir, filename)
                if os.path.exists(gt_path):
                    gt = np.float32(load_img(gt_path)) / 255.

                    # computes PSNR and SSIM
                    psnr_val = psnr(gt, restored, data_range=1.0)
                    ssim_val = ssim(gt, restored, data_range=1.0, channel_axis=2)

                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)

            # saving restored image
            if args.save_images:
                save_path = os.path.join(
                    result_dir, 
                    os.path.splitext(os.path.basename(file_))[0] + '.png'
                )
                save_img(save_path, img_as_ubyte(restored))

    print(f"Results for {dataset_name} ({args.split})")
    print("=" * 50)

    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print(f"PSNR: {avg_psnr:.4f} dB")
        print(f"SSIM: {avg_ssim:.4f}")
    else:
        print("No ground truth available for metric computation")

    if args.save_images:
        print(f"\nRestored images saved to: {result_dir}")


if __name__ == '__main__':
    main()
