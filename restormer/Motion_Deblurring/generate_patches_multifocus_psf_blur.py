"""
Shared-GT patch generator for the defocus deblurring.

* TRAIN (mode=tile, --ps 512 --stride 256) produces the --patch-size 512 --overlap 256 grid.
  NOTE: stride == ps - overlap.
* The clean GT is cropped ONCE into <split>/<group>/shared/target_crops and reused by every level.
* GT<->LQ correspondence is enforced by matching filename + an identical grid, not by sort-order zipping.

Modes
-----
  tile   : overlapping patches (training).          names: <stem>-<i>.png / <stem>.png
  center : one center crop of --crop_size.          names: <stem>.png
  full   : whole image, no crop.                    names: <stem>.png

Output layout  (split -> group -> {shared|level})
-------------------------------------------------
  <out_root>/<split>/<group>/shared/target_crops/...   # written ONCE
  <out_root>/<split>/<group>/<level>/input_crops/...   # per defocus level

Example
-------
# training crops for the Defocus F4 + F6 + F8 All-in-One dataset:
  python generate_patches_multifocus_psf_blur.py \
      --gt_dir  ../../datasets/DIV2K/DIV2K_train_HR \
      --lq_dirs ../../datasets/multifocus_psfs/psf_f4_192_convolved_div2k/train/blur \
                ../../datasets/multifocus_psfs/psf_f6_192_convolved_div2k/train/blur \
                ../../datasets/multifocus_psfs/psf_f8_192_convolved_div2k/train/blur \
      --levels  f4 f6 f8 \
      --out_root ./Datasets --group Defocus \
      --split train --mode tile --ps 512 --stride 256

# validation crops for the Defocus F4 + F6 + F8 All-in-One dataset:
    python generate_patches_multifocus_psf_blur.py \
    --gt_dir  ../../datasets/DIV2K/DIV2K_valid_HR \
    --lq_dirs ../../datasets/multifocus_psfs/psf_f4_192_convolved_div2k/val/blur \
              ../../datasets/multifocus_psfs/psf_f6_192_convolved_div2k/val/blur \
              ../../datasets/multifocus_psfs/psf_f8_192_convolved_div2k/val/blur \
    --levels  f4 f6 f8 \
    --out_root ./Datasets --group Defocus \
    --split val --mode center --crop_size 512
"""
import argparse
import os
from glob import glob

import cv2
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm

DEFAULT_LQ_DIRS = [
    '../../datasets/multifocus_psfs/psf_f0_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f1_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f2_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f3_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f4_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f5_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f6_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f7_192_convolved_div2k',
    '../../datasets/multifocus_psfs/psf_f8_192_convolved_div2k'
]

DEFAULT_LEVELS = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']


def tile_grid(h, w, ps, stride):
    """Exact position set of the old extract_patches (stop-exclusive range + edge append)."""
    if h <= ps and w <= ps:
        return [(0, 0)]    # image <= patch -> single whole crop
    ys = list(range(0, h - ps, stride))
    if (h - ps) not in ys:
        ys.append(h - ps)
    xs = list(range(0, w - ps, stride))
    if (w - ps) not in xs:
        xs.append(w - ps)
    return [(y, x) for y in ys for x in xs]


def make_crops(img, mode, ps, stride, crop_size):
    """Return list of crops in a deterministic (row-major) order."""
    h, w = img.shape[:2]
    if mode == "tile":
        grid = tile_grid(h, w, ps, stride)
        return [img[y:y + ps, x:x + ps] for (y, x) in grid]
    if mode == "center":
        if h >= crop_size and w >= crop_size:
            i, j = (h - crop_size) // 2, (w - crop_size) // 2
            img = img[i:i + crop_size, j:j + crop_size]
        return [img]
    return [img]


def crop_names(stem, n, mode):
    """Suffix only when a tile splits into >1."""
    if mode == "tile" and n > 1:
        return [f"{stem}-{k}.png" for k in range(1, n + 1)]
    return [f"{stem}.png"]


def process_one(name, src_dir, out_dir, mode, ps, stride, crop_size):
    img = cv2.imread(os.path.join(src_dir, name))
    if img is None:
        print(f"Warning: could not read {name} in {src_dir}")
        return 0
    crops = make_crops(img, mode, ps, stride, crop_size)
    stem = os.path.splitext(name)[0]
    for crop, fn in zip(crops, crop_names(stem, len(crops), mode)):
        cv2.imwrite(os.path.join(out_dir, fn), crop)
    return len(crops)


def run_dir(src_dir, out_dir, names, mode, ps, stride, crop_size, workers, desc):
    os.makedirs(out_dir, exist_ok=True)
    fn = lambda nm: process_one(nm, src_dir, out_dir, mode, ps, stride, crop_size)
    if workers > 1:
        counts = Parallel(n_jobs=workers)(delayed(fn)(nm) for nm in tqdm(names, desc=desc))
    else:
        counts = [fn(nm) for nm in tqdm(names, desc=desc)]
    return sum(counts)


def main():
    ap = argparse.ArgumentParser(description="Shared-GT patch generator for the defocus study")
    ap.add_argument("--gt_dir", required=True, help="clean (sharp) source dir for this split")
    ap.add_argument("--lq_dirs", nargs="+", default=DEFAULT_LQ_DIRS, help="one blur source dir per level")
    ap.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS, help="folder names, e.g. f4 f6 f8 (len == lq_dirs)")
    ap.add_argument("--out_root", default="./Datasets", help="output dataset root")
    ap.add_argument("--group", default="Defocus", help="dataset group folder under <split>")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--mode", required=True, choices=["tile", "center", "full"])
    ap.add_argument("--ps", type=int, default=512, help="tile size (mode=tile)")
    ap.add_argument("--stride", type=int, default=256, help="tile step = ps - overlap (mode=tile)")
    ap.add_argument("--crop_size", type=int, default=512, help="center-crop size (mode=center)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert len(args.lq_dirs) == len(args.levels)

    names = [os.path.basename(p) for p in natsorted(glob(os.path.join(args.gt_dir, "*.png")))]
    assert names, f"no .png found in {args.gt_dir}"
    print(f"[{args.split}/{args.group}/{args.mode}] {len(names)} source images")

    base = os.path.join(args.out_root, args.split, args.group)

    # 1) shared GT -- written once
    gt_out = os.path.join(base, "shared", "target_crops")
    n_gt = run_dir(args.gt_dir, gt_out, names, args.mode, args.ps, args.stride,
                   args.crop_size, args.workers, "shared GT")
    print(f"[shared GT] -> {n_gt} files @ {gt_out}")

    # 2) per-level LQ -- identical grid + names => aligned with shared GT
    ref = cv2.imread(os.path.join(args.gt_dir, names[0])).shape[:2]
    for lvl, lq_dir in zip(args.levels, args.lq_dirs):
        lq_dir = os.path.join(lq_dir, args.split, "blur")
        n_lq = len(glob(os.path.join(lq_dir, "*.png")))
        if n_lq == 0:
            print(f"Warning: no .png found in {lq_dir} -- skipped")
            continue
        lq0 = cv2.imread(os.path.join(lq_dir, names[0])).shape[:2]
        assert lq0 == ref, (f"{lvl}: LQ size {lq0} != GT size {ref}; "
                            f"convolution must be same-size and pixel-aligned")
        lq_out = os.path.join(base, lvl, "input_crops")
        n_lq = run_dir(lq_dir, lq_out, names, args.mode, args.ps, args.stride,
                       args.crop_size, args.workers, lvl)
        assert n_lq == n_gt, f"{lvl}: file count {n_lq} != GT {n_gt} (alignment broken)"
        print(f"[{lvl}] -> {n_lq} files @ {lq_out}")


if __name__ == "__main__":
    main()