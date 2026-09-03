"""
Validation / sanity checks for a PSF capture session before merging.

Checks, per field position and exposure:
  - File format: dtype, shape, bit depth, whether data is raw Bayer
    (single channel) or demosaiced (3 identical or near-identical channels).
  - Saturation: fraction of pixels at or above the saturation threshold in
    the signal frames (core should saturate only at the long exposures).
  - Dark level and read-noise proxy from the dark stack.
  - Glare pedestal: mean of (glare - dark). If your blackout is good this is
    near zero; if not, this is the veiling-glare level that must be removed.
  - Blackout check: reports whether glare mean significantly exceeds dark
    mean (the quantitative version of the live-feed eyeball check).

Usage:
    python psf_validation.py --root ../psf_multifocus_sweep/F4
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
        return np.load(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, '*')))
    files = [f for f in files if f.lower().endswith(('.npy', '.png'))]
    if not files:
        return None, []
    frames = [load_frame(f).astype(np.float64) for f in files]
    return np.stack(frames, axis=0), files


def to_mono(arr):
    """Collapse a demosaiced 3-channel frame to single channel for stats.
    If already single-channel, return as is."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # If channels identical, any channel works; else average.
        return arr.mean(axis=-1)
    raise ValueError(f"unexpected shape {arr.shape}")


def describe_format(sample):
    info = {
        'dtype': str(sample.dtype),
        'shape': sample.shape,
        'min': float(sample.min()),
        'max': float(sample.max()),
    }
    if sample.ndim == 3:
        ch_identical = all(np.array_equal(sample[..., 0], sample[..., i]) 
                           for i in range(1, sample.shape[-1]))
        info['channels_identical'] = ch_identical
        info['likely'] = ('demosaiced (grayscale replicated to 3ch)'
                          if ch_identical else 'demosaiced (true color)')
    else:
        info['likely'] = 'raw Bayer mosaic (single channel)'
    return info


def analyze_exposure(exp_dir):
    out = {}
    dark, df = load_stack(os.path.join(exp_dir, 'dark'))
    glare, gf = load_stack(os.path.join(exp_dir, 'glare'))
    signal, sf = load_stack(os.path.join(exp_dir, 'signal'))

    if signal is None:
        return None
    out['n_dark'] = 0 if dark is None else len(df)
    out['n_glare'] = 0 if glare is None else len(gf)
    out['n_signal'] = len(sf)

    sig_mean = to_mono(signal.mean(axis=0))
    out['signal_max'] = float(sig_mean.max())
    out['signal_sat_frac'] = float((sig_mean >= 255 * 0.999).mean())

    if dark is not None:
        dk_mono = to_mono(dark.mean(axis=0))
        out['dark_mean'] = float(dk_mono.mean())
        # Read-noise proxy: temporal std across dark frames, averaged over pixels.
        dk_temporal_std = to_mono(dark.std(axis=0)).mean()
        out['dark_temporal_std'] = float(dk_temporal_std)
    else:
        out['dark_mean'] = None

    if glare is not None and dark is not None:
        gl_mono = to_mono(glare.mean(axis=0))
        dk_mono = to_mono(dark.mean(axis=0))
        pedestal = (gl_mono - dk_mono)
        out['glare_minus_dark_mean'] = float(pedestal.mean())
        out['glare_minus_dark_p99'] = float(np.percentile(pedestal, 99))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, required=True,
                    help="A field-position dir containing exposure_* subdirs.")
    args = ap.parse_args()

    exp_dirs = sorted(d for d in glob.glob(os.path.join(str(args.root), 'exposure_*')) if os.path.isdir(d))
    if not exp_dirs:
        sys.exit(f"No exposure_* directories under {args.root}")

    # Format check on one sample.
    sample_files = glob.glob(os.path.join(exp_dirs[0], 'signal', '*'))
    sample = load_frame(sorted(sample_files)[0])
    fmt = describe_format(sample)
    print("\nFORMAT CHECK (from first signal frame)")
    print("-" * 40)
    for k, v in fmt.items():
        print(f"{k}: {v}")

    print(f"\nPER-EXPOSURE ANALYSIS")
    print("-" * 40)
    for ed in exp_dirs:
        name = os.path.basename(ed)
        r = analyze_exposure(ed)
        if r is None:
            print(f"\n{name}: no signal frames found"); continue
        print(f"\n{name}:")
        print(f"frames (dark/glare/signal): "
              f"{r['n_dark']}/{r['n_glare']}/{r['n_signal']}")
        print(f"signal max DN: {r['signal_max']:.1f}")
        print(f"signal saturated frac: {r['signal_sat_frac']*100:.3f}%")
        if r.get('dark_mean') is not None:
            print(f"dark mean DN: {r['dark_mean']:.2f}")
            print(f"dark temporal std: {r['dark_temporal_std']:.2f} (read-noise proxy)")
        if 'glare_minus_dark_mean' in r:
            gm = r['glare_minus_dark_mean']
            print(f"glare - dark mean: {gm:+.3f} DN")
            print(f"glare - dark p99: {r['glare_minus_dark_p99']:+.3f} DN")
            # Blackout verdict
            noise = r.get('dark_temporal_std', 1.0)
            if abs(gm) < 0.5 * noise:
                print(f" -> blackout GOOD (pedestal below noise floor)")
            elif abs(gm) < 3 * noise:
                print(f" -> minor pedestal; glare subtraction will handle it")
            else:
                print(f" -> SIGNIFICANT pedestal ({gm:.1f} DN).")
                print(f"Glare subtraction is essential; consider improving blackout.")

    print("\n" + "-" * 40)
    print("Interpretation:")
    print("- Short exposure should have LOW saturated frac (clean core)")
    print("- Long exposure MAY saturate the core (wings are the target there)")
    print("- Glare - Dark should be near zero if blackout is good")


if __name__ == '__main__':
    main()
