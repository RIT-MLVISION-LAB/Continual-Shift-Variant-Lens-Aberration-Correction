"""Build a pooled meta-info file for all-in-one training over the six blur
variants produced by run_blur_synthesis_pipeline.py.

Layout expected (dataset mode of the synthesis pipeline):
    <root>/<split>/variant_k/input_crops/*.png
    <root>/<split>/variant_k/target_crops/*.png

Usage:
    python build_aio_meta_info.py --root ./Datasets --split train \
        --out  ./Datasets/meta_aio_train.txt
    python build_aio_meta_info.py --root ./Datasets --split val \
        --out  ./Datasets/meta_aio_val.txt
"""
import argparse
import os
import os.path as osp

EXTS = ('.png', '.jpg', '.jpeg')


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-variants', type=int, default=6)
    args = ap.parse_args()

    lines, counts = [], []
    for k in range(1, args.num_variants + 1):
        blur_dir = osp.join(args.root, args.split, f'ShiftVariant_V{k}', 'input_crops')
        sharp_dir = osp.join(args.root, args.split, f'ShiftVariant_V{k}', 'target_crops')
        blur_imgs = list_images(blur_dir)
        sharp_set = set(list_images(sharp_dir))
        n = 0
        for name in blur_imgs:
            if name not in sharp_set:
                raise FileNotFoundError(f'no GT for {blur_dir}/{name}')
            lq_rel = osp.join(args.split, f'ShiftVariant_V{k}', 'input_crops', name)
            gt_rel = osp.join(args.split, f'ShiftVariant_V{k}', 'target_crops', name)
            lines.append(f'{lq_rel} {gt_rel} {k - 1}')
            n += 1
        counts.append(n)
        print(f'variant_{k}: {n} pairs')

    if len(set(counts)) != 1:  # assert equal image counts across variants
        raise ValueError(f'variant counts differ: {counts}')

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {len(lines)} lines -> {args.out}')


if __name__ == '__main__':
    main()