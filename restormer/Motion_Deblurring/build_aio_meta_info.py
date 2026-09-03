"""
Build a pooled meta-info file for all-in-one training over the defocus blur
levels, with a SHARED ground-truth folder.

Layout (matches the Defocus_*_{Restormer,PromptIR}.yml configs):
    <root>/<split>/<group>/<level>/input_crops/*.png     # per-level blur (lq)
    <root>/<split>/<group>/shared/target_crops/*.png     # shared GT (one copy)

Emitted line (parsed by Dataset_PairedAiO -> lq_rel gt_rel int_domain):
    train/Defocus/f4/input_crops/0001.png train/Defocus/shared/target_crops/0001.png 0

Usage:
    python build_aio_meta_info.py --root ./Datasets --split train \
        --out ./Datasets/meta_aio_defocus_train.txt
    python build_aio_meta_info.py --root ./Datasets --split val \
        --out ./Datasets/meta_aio_defocus_val.txt
"""
import argparse
import os
import os.path as osp

EXTS = ('.png', '.jpg', '.jpeg')


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help="== dataset 'dataroot' in the AiO YAML")
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--group', default='Defocus')
    ap.add_argument('--levels', nargs='+', default=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
                    help='folder names; index in this list is the domain id')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gt_rel_dir = osp.join(args.split, args.group, 'shared', 'target_crops')
    gt_set = set(list_images(osp.join(args.root, gt_rel_dir)))

    lines, counts = [], []
    for dom, lvl in enumerate(args.levels):
        lq_rel_dir = osp.join(args.split, args.group, lvl, 'input_crops')
        imgs = list_images(osp.join(args.root, lq_rel_dir))
        for name in imgs:
            if name not in gt_set:
                raise FileNotFoundError(f'no shared GT for {lvl}/{name}')
            lines.append(f'{osp.join(lq_rel_dir, name)} '
                         f'{osp.join(gt_rel_dir, name)} {dom}')
        counts.append(len(imgs))
        print(f'{lvl}: {len(imgs)} pairs (domain {dom})')

    if len(set(counts)) != 1:
        raise ValueError(f'level counts differ: {counts}')

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {len(lines)} lines -> {args.out}')


if __name__ == '__main__':
    main()