"""
Convolve DIV2K (train+val) with resized through-focus PSFs to produce
BLUR-ONLY outputs for a paired-dataset workflow. Sharp GT is NOT written. 
The original DIV2K directory is the shared sharp source.

Output layout per defocus level (blur only):
    <out-root>/psf_<lvl>_<size>_convolved_div2k/
        train/blur/
        val/blur/

Convolution is per-channel in linear radiance (sRGB linearized, convolved,
re-encoded) so the blur is physically correct.

Parallelism: CPU process pool. The workload is imread -> fftconvolve -> imwrite. 
Levels are processed concurrently; within a level, images are sharded across the same pool.
Use --workers to size the pool (default: os.cpu_count()).

Usage:
    python psf_convolve.py \\
        --train ../../Continual-Aberration-Correction/datasets/DIV2K/DIV2K_train_HR/ \\
        --val ../../Continual-Aberration-Correction/datasets/DIV2K/DIV2K_valid_HR/ \\
        --psf-dir multifocus_psfs/psf_192/ --levels f0 f1 f2 f3 f4 f5 f6 f7 f8 \\
        --out-root ../../Continual-Aberration-Correction/datasets/multifocus_psfs/ \\
        --size 192 --workers 9

    # single level:
    python psf_convolve.py --train ... --val ... \\
        --psf psf_192/psf_f0_192.npy --lvl f0 --out-root datasets/ --size 192
"""
import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from scipy.signal import fftconvolve

# Keep each worker single-threaded for BLAS/OpenCV so N workers use N cores
# cleanly instead of oversubscribing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass


def srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x):
    a = 0.055
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


def prep_psf(psf, n_ch):
    if psf.ndim == 2:
        planes = [psf] * n_ch
    elif psf.ndim == 3 and psf.shape[-1] == n_ch:
        planes = [psf[..., i] for i in range(n_ch)]
    else:
        g = psf.mean(axis=-1) if psf.ndim == 3 else psf
        planes = [g] * n_ch
    out = []
    for p in planes:
        p = np.clip(p.astype(np.float64), 0, None)
        s = p.sum()
        out.append(p / s if s > 0 else p)
    return out


def blur_image(srgb01, psf_planes):
    lin = srgb_to_linear(srgb01)
    out = np.empty_like(lin)
    kh, kw = psf_planes[0].shape
    py, px = kh // 2, kw // 2
    for c in range(lin.shape[-1]):
        padded = np.pad(lin[..., c], ((py, py), (px, px)), mode='reflect')
        conv = fftconvolve(padded, psf_planes[c], mode='same')
        out[..., c] = conv[py:py + lin.shape[0], px:px + lin.shape[1]]
    return linear_to_srgb(out)


def list_images(d):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in exts)


# ---- worker: one (level, image) job -----------------------------------------
def _convolve_one(args):
    """Blur a single image for a single level. Returns (lvl, name, status)."""
    img_path, blur_dir, psf_path, png_level = args
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return (blur_dir, img_path.name, "skip")
    # Small (192x192x3) PSF is loaded per job.
    # So overhead is negligible and keeps workers stateless.
    psf = np.load(psf_path).astype(np.float64)
    planes = prep_psf(psf, img.shape[-1])
    srgb01 = img.astype(np.float64) / 255.0
    out01 = blur_image(srgb01, planes)
    out = np.clip(np.round(out01 * 255.0), 0, 255).astype(np.uint8)
    Path(blur_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(blur_dir) / img_path.name), out,
                [cv2.IMWRITE_PNG_COMPRESSION, png_level])
    return (blur_dir, img_path.name, "ok")


def build_jobs(levels_psf, train_imgs, val_imgs, out_root, size, png_level):
    """Flatten all (level, split, image) into a single job list so the pool
    stays saturated across levels rather than finishing one before the next."""
    jobs = []
    for lvl, psf_path in levels_psf:
        ds = out_root / f"psf_{lvl}_{size}_convolved_div2k"
        for split_imgs, split in [(train_imgs, 'train'), (val_imgs, 'val')]:
            blur_dir = ds / split / 'blur'
            for p in split_imgs:
                jobs.append((p, str(blur_dir), str(psf_path), png_level))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--val', type=Path, required=True)
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count(),
                    help="Process pool size (CPU). Default: all cores.")
    ap.add_argument('--png-level', type=int, default=3,
                    help="PNG compression 0-9 for blur outputs.")
    # single-level
    ap.add_argument('--psf', type=Path, default=None)
    ap.add_argument('--lvl', type=str, default=None)
    # multi-level
    ap.add_argument('--psf-dir', type=Path, default=None)
    ap.add_argument('--levels', nargs='+', default=None)
    args = ap.parse_args()

    train_imgs = list_images(args.train)
    val_imgs = list_images(args.val)
    if not train_imgs or not val_imgs:
        sys.exit("empty train or val directory")
    print(f"DIV2K: {len(train_imgs)} train, {len(val_imgs)} val")
    print(f"Sharp GT: using original DIV2K as shared source (no sharp written)")
    print(f"Workers: {args.workers} (CPU process pool)")

    # Resolve (level -> psf path)
    levels_psf = []
    if args.psf and args.lvl:
        levels_psf.append((args.lvl, args.psf))
    elif args.psf_dir and args.levels:
        for lvl in args.levels:
            pf = args.psf_dir / f"psf_{lvl}_{args.size}.npy"
            if not pf.exists():
                print(f"[warn] missing {pf}, skipping {lvl}")
                continue
            levels_psf.append((lvl, pf))
    else:
        sys.exit("give (--psf and --lvl) or (--psf-dir and --levels)")

    jobs = build_jobs(levels_psf, train_imgs, val_imgs,
                      args.out_root, args.size, args.png_level)
    total = len(jobs)
    print(f"Levels: {[l for l, _ in levels_psf]}  |  total blur jobs: {total}")

    done = 0
    fails = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_convolve_one, j) for j in jobs]
        for fut in as_completed(futs):
            _, name, status = fut.result()
            done += 1
            if status != "ok":
                fails += 1
            if done % 200 == 0 or done == total:
                print(f"  {done}/{total} done ({fails} skipped/failed)")

    print(f"\nDone. {total - fails} blur images written across {len(levels_psf)} levels, {fails} skipped.")


if __name__ == '__main__':
    main()
