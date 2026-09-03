"""
Resize cleaned+cropped through-focus PSFs to a fixed support size.

Reads per-defocus-level kernels (naming: psf_f<N>_clean_crop_768.npy),
resizes each to the target size with area-based downsampling, renormalizes 
each to sum 1 independently, and saves as psf_f<N>_<size>.npy.

Visualization: a grid of all resized levels (linear, each panel
normalized to its own peak) so the through-focus progression is visible
at a glance.

Usage:
    # all levels:
    python psf_resize.py --in-dir multifocus_psfs/ --levels f0 f1 f2 f3 f4 f5 f6 f7 f8 \\
        --size 192 --out-dir multifocus_psfs/psf_192/ --viz

    # single level:
    python psf_resize.py --in-dir multifocus_psfs/ --levels f4 \\
        --size 192 --out-dir multifocus_psfs/psf_192/
"""
import argparse
import math
from pathlib import Path

import numpy as np
import cv2


def resize_psf(psf, size):
    if psf.ndim == 2:
        r = cv2.resize(psf.astype(np.float64), (size, size),
                       interpolation=cv2.INTER_AREA)
        r = np.clip(r, 0, None)
        s = r.sum()
        return (r / s if s > 0 else r).astype(np.float32)
    out = np.empty((size, size, psf.shape[-1]), np.float64)
    for c in range(psf.shape[-1]):
        rc = cv2.resize(psf[..., c].astype(np.float64), (size, size),
                        interpolation=cv2.INTER_AREA)
        rc = np.clip(rc, 0, None)
        s = rc.sum()
        out[..., c] = rc / s if s > 0 else rc
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--in-dir', type=Path, required=True,
                    help="Directory holding psf_f<N>_clean_crop_768.npy files.")
    ap.add_argument('--levels', nargs='+', required=True,
                    help="Defocus level tags, e.g. f0 f1 ... f8.")
    ap.add_argument('--size', type=int, default=192,
                    help="Target support size (e.g. 192).")
    ap.add_argument('--src-size', type=int, default=768,
                    help="Source crop size in the input filenames.")
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--viz', action='store_true')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resized = {}
    for lvl in args.levels:
        inp = args.in_dir / f"psf_{lvl}_clean_crop_{args.src_size}.npy"
        if not inp.exists():
            print(f"[warn] missing {inp}, skipping")
            continue
        psf = np.load(inp).astype(np.float64)
        r = resize_psf(psf, args.size)
        outp = args.out_dir / f"psf_{lvl}_{args.size}.npy"
        np.save(outp, r)
        e = r if r.ndim == 2 else r.sum(axis=-1)
        resized[lvl] = e
        print(f"  {lvl}: {psf.shape} -> {r.shape}  sum={e.sum():.4f}  peak={e.max():.4g}  -> {outp}")

    if args.viz and resized:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(resized)
        ncol = min(3, n)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis('off')
        for ax, (lvl, e) in zip(axes, resized.items()):
            ax.imshow(e / (e.max() + 1e-12), cmap='inferno', vmin=0, vmax=1)
            ax.set_title(lvl.upper(), fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        vp = f"viz/psf_resized_grid_{args.size}.png"
        fig.savefig(vp, dpi=140, bbox_inches='tight')
        print(f"Saved resized grid -> {vp}")


if __name__ == '__main__':
    main()
