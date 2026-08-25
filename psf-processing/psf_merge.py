"""
Merge a bracketed PSF capture into a single normalized blur kernel.

Pipeline per exposure:
  1. Average the dark stack; average the glare stack; average the signal stack.
  2. Dark-subtract signal and glare.
  3. Glare (veiling) removal: subtract (glare - dark) from (signal - dark).
     This removes the broad stray-light pedestal while leaving the PSF.
  4. Convert to linear radiance by dividing by exposure time (per-exposure
     scaling). All exposures now share a common radiance scale.
  5. Saturation masking: pixels at/above sat_dn (255) in the RAW signal average are
     unreliable (clipped) and excluded from that exposure's contribution.
Merge across exposures:
  6. Weighted combination in radiance space. Each pixel's value is the
     weighted mean over the exposures where it is neither saturated nor
     noise-dominated; long exposures dominate the wings, short exposures
     provide the (unclipped) core.
Finalize:
  7. Clip negatives to zero (radiance is non-negative; residual negatives are
     noise after subtraction).
  9. Energy normalization so the kernel sums to 1 (required for a
     brightness-preserving convolution).

Usage:
    python psf_merge.py --root ../psf_multifocus_sweep/F4 \\
        --exposures 20000 80000 320000 \\
        --out multifocus_psfs/psf_f4.npy --viz viz/psf_f4_preview.png
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import cv2


def load_frame(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float64)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64)


def mean_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, '*')))
    files = [f for f in files if f.lower().endswith(('.npy', '.png', '.tif', '.tiff'))]
    if not files:
        return None
    acc = None
    for f in files:
        fr = load_frame(f)
        acc = fr if acc is None else acc + fr
    return acc / len(files)


def to_working(arr):
    """Return a dict of named planes to process independently."""
    if arr.ndim == 2:
        return {'mono': arr}
    return {f'c{i}': arr[..., i] for i in range(arr.shape[-1])}


def process(args):
    exps = args.exposures
    print(f"Exposures (us): {exps}")

    # Load and reduce every stack up front.
    per_exp = {}
    for t in exps:
        ed = os.path.join(str(args.root), f"exposure_{t}us")
        if not os.path.isdir(ed):
            sys.exit(f"Missing exposure dir: {ed}")
        dark = mean_stack(os.path.join(ed, 'dark'))
        glare = mean_stack(os.path.join(ed, 'glare'))
        signal = mean_stack(os.path.join(ed, 'signal'))
        if signal is None:
            sys.exit(f"No signal frames in {ed}")
        if dark is None:
            print(f"[warn] no dark for exposure_{t}us; assuming zero dark")
            dark = np.zeros_like(signal)
        if glare is None:
            print(f"[warn] no glare for exposure_{t}us; skipping glare removal")
            glare = dark.copy()
        per_exp[t] = dict(dark=dark, glare=glare, signal=signal)

    # Determine planes from the first signal frame
    sample = per_exp[exps[0]]['signal']
    plane_names = list(to_working(sample).keys())
    print(f"Planes: {plane_names}")

    merged_planes = {}
    for pname in plane_names:
        # Accumulators for weighted merge in radiance space
        num = None  # sum of weight * radiance
        den = None  # sum of weight

        longest_t = max(exps)
        for t in exps:
            d = per_exp[t]
            sig = to_working(d['signal'])[pname]
            drk = to_working(d['dark'])[pname]
            gl  = to_working(d['glare'])[pname]

            # Saturation mask from RAW signal (before subtraction)
            unsat = sig < (255 * 0.999)

            # Dark subtract, then glare (veiling) removal
            corrected = (sig - drk) - (gl - drk)  # signal - glare, but explicit

            # To linear radiance: divide by (effective) integration time.
            eff_t = t
            radiance = corrected / eff_t

            # Weight: prefer well-exposed pixels. Down-weight near-saturated
            # and near-noise. Simple, robust triangular weight on raw DN.
            dn = sig
            w = np.minimum(dn - drk, 255 - dn)
            w = np.clip(w, 0, None)
            w = w * unsat

            contrib_num = w * radiance
            contrib_den = w
            num = contrib_num if num is None else num + contrib_num
            den = contrib_den if den is None else den + contrib_den

        # Avoid divide-by-zero: pixels unseen at any exposure -> 0
        radiance_merged = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        radiance_merged = np.clip(radiance_merged, 0, None)
        merged_planes[pname] = radiance_merged

    # Stack planes into a kernel array
    if len(merged_planes) == 1:
        kernel = next(iter(merged_planes.values()))
    else:
        kernel = np.stack([merged_planes[n] for n in plane_names], axis=-1)

    energy = kernel if kernel.ndim == 2 else kernel.sum(axis=-1)

    # --- energy normalization (sum to 1, per channel if multi-channel) ---
    if kernel.ndim == 2:
        s = kernel.sum()
        if s > 0:
            kernel = kernel / s
    else:
        for i in range(kernel.shape[-1]):
            s = kernel[..., i].sum()
            if s > 0:
                kernel[..., i] = kernel[..., i] / s

    np.save(args.out, kernel.astype(np.float32))
    print(f"Saved kernel {kernel.shape} dtype float32 -> {args.out}")
    print(f"sum(s): "
          + (f"{kernel.sum():.4f}" if kernel.ndim == 2
             else ", ".join(f"{kernel[...,i].sum():.4f}" for i in range(kernel.shape[-1]))))

    # --- preview PNG (log-stretched for visibility of wings) ---
    if args.viz:
        vis = energy if 'energy' in dir() else (
            kernel if kernel.ndim == 2 else kernel.sum(axis=-1))
        vis = kernel.sum(axis=-1) if kernel.ndim == 3 else kernel
        v = vis / (vis.max() + 1e-12)
        v_log = np.log1p(v * 1000) / np.log1p(1000)
        img = (np.clip(v_log, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(args.viz), img)
        print(f"Saved log-stretched preview -> {args.viz}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--exposures', type=int, nargs='+', required=True,
                    help="Exposure times in us, matching exposure_<t>us dir names.")
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--viz', type=Path, default=None)
    args = ap.parse_args()
    process(args)


if __name__ == '__main__':
    main()
