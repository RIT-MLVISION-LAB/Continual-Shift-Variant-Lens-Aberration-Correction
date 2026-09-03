"""
Crop PSFs to a fixed window and produce two figures:
  1. Side-by-side linear-scale PSFs (true scale, not log) so the actual
     convolution kernels are shown as they are.
  2. Encircled-energy ring diagram: concentric circles at chosen energy
     fractions, colored by cumulative energy, annotated with per-ring and
     cumulative energy, to justify the crop radius.

Usage:
    python psf_crop_and_viz.py --save-cropped
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

DEFAULT_PSF_PATHS = [
    "multifocus_psfs/psf_f0_clean.npy",
    "multifocus_psfs/psf_f1_clean.npy",
    "multifocus_psfs/psf_f2_clean.npy",
    "multifocus_psfs/psf_f3_clean.npy",
    "multifocus_psfs/psf_f4_clean.npy",
    "multifocus_psfs/psf_f5_clean.npy",
    "multifocus_psfs/psf_f6_clean.npy",
    "multifocus_psfs/psf_f7_clean.npy",
    "multifocus_psfs/psf_f8_clean.npy",
]

DEFAULT_LABELS = [
    "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"
]


def energy(psf):
    return psf if psf.ndim == 2 else psf.sum(axis=-1)


def centroid(e):
    ys, xs = np.indices(e.shape)
    t = e.sum()
    return (xs * e).sum() / t, (ys * e).sum() / t  # (cx, cy)


def center_crop(psf, size):
    e = energy(psf)
    cx, cy = centroid(e)
    cx, cy = int(round(cx)), int(round(cy))
    half = size // 2
    pad = half + 5
    if psf.ndim == 2:
        p = np.pad(psf, pad)
    else:
        p = np.pad(psf, ((pad, pad), (pad, pad), (0, 0)))
    cx += pad; cy += pad
    return p[cy - half:cy + half, cx - half:cx + half]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--psfs', nargs='+', default=DEFAULT_PSF_PATHS,
                    help="PSF .npy files to crop and show. Provide them in the "
                         "spatial order you want them displayed left-to-right "
                         "(e.g. left center right).")
    ap.add_argument('--labels', nargs='+', default=DEFAULT_LABELS)
    ap.add_argument('--size', type=int, default=768)
    ap.add_argument('--save-cropped', action='store_true',
                    help="Save each cropped PSF as <name>_crop_<args.size>.npy.")
    ap.add_argument('--grid', default='viz/psf_grid_linear.png')
    ap.add_argument('--rings', default='viz/psf_energy_rings.png',
                    help="Ring diagram; uses the FIRST psf given.")
    ap.add_argument('--gamma', type=float, default=1.0,
                    help="Display gamma for the side-by-side ONLY (1.0 = true "
                         "linear). Does not affect saved kernels.")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.psfs]

    cropped = []
    for p in args.psfs:
        psf = np.load(p).astype(np.float64)
        c = center_crop(psf, args.size)
        # renormalize the crop to unit energy (this is the actual kernel)
        e = energy(c)
        c = c / (e.sum() + 1e-12)
        cropped.append(c)
        if args.save_cropped:
            outp = Path(p).with_name(Path(p).stem + f"_crop_{args.size}.npy")
            np.save(outp, c.astype(np.float32))
            print(f"Saved cropped kernel -> {outp}")

    # ---- Figure 1: side-by-side, linear (shared scale) ----
    n = len(cropped)
    emaps = [energy(c) for c in cropped]
    vmax = max(e.max() for e in emaps)   # shared scale so they're comparable
    fig, axes = plt.subplots(n // 3, n // 3, figsize=(5 * (n // 3), 5 * (n // 3)))
    axes = axes.ravel() if n > 1 else [axes]
    if n == 1:
        axes = [axes]
    for ax, e, lab in zip(axes, emaps, labels):
        disp = e / vmax
        if args.gamma != 1.0:
            disp = np.power(np.clip(disp, 0, 1), args.gamma)
        ax.imshow(disp, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(lab, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.grid, dpi=300, bbox_inches='tight')
    print(f"Saved PSF grid -> {args.grid}")

    # ---- Figure 2: encircled-energy ring diagram ----
    # Default to the middle PSF
    ridx = len(emaps) // 2
    e0 = emaps[ridx]
    e0 = np.clip(e0, 0, None)
    H, W = e0.shape
    cy, cx = H / 2.0, W / 2.0
    ys, xs = np.indices(e0.shape)
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    tot = e0.sum()

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    cum_sorted = np.cumsum(e0.ravel()[order]) / tot

    fracs = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    radii = []
    for f in fracs:
        idx = np.searchsorted(cum_sorted, f)
        radii.append(r_sorted[min(idx, len(r_sorted) - 1)])

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    disp = e0 / e0.max()
    disp = np.power(disp, 0.5)  # mild gamma just for the backdrop visibility
    ax2.imshow(disp, cmap='gray', vmin=0, vmax=1)

    colors = cm.viridis(np.linspace(0.15, 0.95, len(fracs)))
    prev_cum = 0.0
    for f, rr, col in zip(fracs, radii, colors):
        ring_pct = (f - prev_cum) * 100
        circ = Circle((cx, cy), rr, fill=False, edgecolor=col, lw=2.0,
                      label=f"r={rr:.0f}px: +{ring_pct:.0f}% (cumulative {f*100:.0f}%)")
        ax2.add_patch(circ)
        prev_cum = f

    ax2.plot(cx, cy, '+', color='red', markersize=10, mew=1.5)
    ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"Circles at cumulative-energy fractions: \n+x% = energy added by that ring")
    leg = ax2.legend(loc='center right', fontsize=8, framealpha=0.6,
                     facecolor='black', edgecolor='white',
                     title="Encircled-energy rings", title_fontsize=9)
    for txt in leg.get_texts():
        txt.set_color('white')
    leg.get_title().set_color('white')
    fig2.tight_layout()
    fig2.savefig(args.rings, dpi=300, bbox_inches='tight')
    print(f"Saved ring diagram -> {args.rings}")

    print(f"\nEncircled-energy radii ({labels[ridx]}):")
    prev = 0.0
    for f, rr in zip(fracs, radii):
        print(f"{f*100:4.0f}% cumulative -> r = {rr:6.1f}px (this ring adds {(f-prev)*100:.0f}%)")
        prev = f


if __name__ == '__main__':
    main()
