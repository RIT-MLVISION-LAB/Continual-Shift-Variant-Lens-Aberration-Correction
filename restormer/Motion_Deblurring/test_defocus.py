"""
Cross-domain / OOD testing for the defocus study.

Uses the SHARED ground-truth folder. Layout (matches the training/val configs):
    <data_root>/<split>/<group>/<level>/input_crops/*.png   # blurred input (lq)
    <data_root>/<split>/<group>/shared/target_crops/*.png   # shared GT (one copy)

Example -- f4 Restormer expert swept across the whole defocus family, one row of
the cross-generalization matrix:
    PYTHONPATH=.. python test_defocus_cropped.py \
        --config ./Options/Defocus_F4_Restormer.yml \
        --weights ../experiments/Defocus_F4_Restormer/models/net_g_latest.pth \
        --data_root ./Datasets/ --tag Restormer_F4 \
        --test_levels f0 f1 f2 f3 f4 f5 f6 f7 f8 \
        --out_csv ./defocus_results/defocus_matrix_f4_restormer.csv
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from natsort import natsorted
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.promptIR_arch import PromptIR

ARCHS = {'Restormer': Restormer, 'PromptIR': PromptIR}


def load_img(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def save_img(p, img):
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def center_crop(img, size):
    if size is None:
        return img
    h, w = img.shape[:2]
    if h <= size and w <= size:
        return img
    i, j = (h - size) // 2, (w - size) // 2
    return img[i:i + size, j:j + size]


def load_config(path):
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def build_model(config_path, weights):
    net = load_config(config_path)['network_g'].copy()
    arch = net.pop('type')
    if arch not in ARCHS:
        raise ValueError(f'network_g.type={arch} not in {list(ARCHS)}')
    model = ARCHS[arch](**net)
    ckpt = torch.load(weights, map_location='cpu')
    state = ckpt.get('params', ckpt.get('params_ema', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}  # tolerate DataParallel prefix
    model.load_state_dict(state, strict=True)
    print(f'[{arch}] loaded {weights}')
    return model, arch


def evaluate_level(model, inp_dir, gt_dir, factor, crop=None, save_dir=None):
    files = natsorted(glob(os.path.join(inp_dir, '*.png')))
    if not files:
        print(f'  (no images in {inp_dir}) -- skipped')
        return None
    ps, ss = [], []
    with torch.no_grad():
        for f_ in tqdm(files, leave=False):
            img = center_crop(np.float32(load_img(f_)) / 255., crop)
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
            h, w = x.shape[2], x.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            x = F.pad(x, (0, padw, 0, padh), 'reflect')

            out = model(x)[:, :, :h, :w]
            out = torch.clamp(out, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            gt = center_crop(np.float32(load_img(os.path.join(gt_dir, os.path.basename(f_)))) / 255., crop)
            ps.append(psnr(gt, out, data_range=1.0))
            ss.append(ssim(gt, out, data_range=1.0, channel_axis=2))
            if save_dir:
                save_img(os.path.join(save_dir, os.path.basename(f_)), img_as_ubyte(out))
    return float(np.mean(ps)), float(np.mean(ss)), len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML with network_g (expert or AiO)')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data_root', default='./Datasets')
    ap.add_argument('--group', default='Defocus')
    ap.add_argument('--split', default='val')
    ap.add_argument('--test_levels', nargs='+', required=True, help='e.g. f0 f2 f4 f5 f6 f7 f8')
    ap.add_argument('--tag', required=True, help='row label for the CSV, e.g. Restormer_expert_f4')
    ap.add_argument('--crop', type=int, default=None,
                    help='center-crop side length for inference (default: full image).')
    ap.add_argument('--save_images', action='store_true')
    ap.add_argument('--result_dir', default='./defocus_results')
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    model, arch = build_model(args.config, args.weights)
    model.cuda().eval()
    factor = 8

    gt_dir = os.path.join(args.data_root, args.split, args.group, 'shared', 'target_crops')
    rows = []
    for lvl in args.test_levels:
        inp_dir = os.path.join(args.data_root, args.split, args.group, lvl, 'input_crops')
        save_dir = None
        if args.save_images:
            save_dir = os.path.join(args.result_dir, args.tag, lvl)
            os.makedirs(save_dir, exist_ok=True)
        res = evaluate_level(model, inp_dir, gt_dir, factor, crop=args.crop, save_dir=save_dir)
        if res is None:
            continue
        p, s, n = res
        crop_tag = args.crop if args.crop else 'full'
        print(f'{args.tag:26} | {lvl:4} | crop={crop_tag} | PSNR {p:7.4f} | SSIM {s:.4f} | n={n}')
        rows.append((args.tag, arch, lvl, crop_tag, f'{p:.4f}', f'{s:.4f}', n))

    if args.out_csv and rows:
        new = not os.path.exists(args.out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, 'a', newline='') as f:
            wtr = csv.writer(f)
            if new:
                wtr.writerow(['tag', 'arch', 'test_level', 'crop', 'psnr', 'ssim', 'n'])
            wtr.writerows(rows)
        print(f'appended {len(rows)} rows -> {args.out_csv}')


if __name__ == '__main__':
    main()