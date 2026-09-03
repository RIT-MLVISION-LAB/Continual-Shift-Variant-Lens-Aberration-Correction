"""
Estimate and subtract a residual pedestal (glare/noise floor) from a PSF,
then re-run encircled-energy analysis to find the true support radius.

A flat marginal-energy-per-ring plateau at large radius is not PSF wing
structure -- it is a uniform floor (residual veiling glare, or read-noise
rectified positive by clip-to-zero after subtraction). Integrating it
inflates the encircled-energy radii and, if baked into a normalized
kernel, adds a broad haze to every convolved image. This script removes it.

Usage:
    python psf_floor_removal.py --psf ./multifocus_psfs/psf_f4.npy --out ./multifocus_psfs/psf_f4_clean.npy
"""
import argparse
import numpy as np


def energy(psf):
    return psf if psf.ndim == 2 else psf.sum(axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--psf', required=True)
    ap.add_argument('--floor-radius', type=float, default=1900,
                    help="Estimate the floor from pixels beyond this radius.")
    ap.add_argument('--out', default=None,
                    help="Save the floor-subtracted, renormalized PSF here.")
    args = ap.parse_args()

    psf = np.load(args.psf).astype(np.float64)
    e = energy(psf)
    tot0 = e.sum()

    ys, xs = np.indices(e.shape)
    cy = (ys * e).sum() / tot0
    cx = (xs * e).sum() / tot0
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    far = r > args.floor_radius
    floor = np.median(e[far])
    floor_frac = floor * e.size / tot0
    print(f"Centroid: ({cx:.1f}, {cy:.1f})")
    print(f"Floor estimate (median beyond r={args.floor_radius:.0f}): {floor:.4g} per pixel")
    print(f"Total floor energy if spread over whole array: {floor_frac*100:.2f}% of raw total")

    # Subtract floor, clip, recompute.
    e_sub = np.clip(e - floor, 0, None)
    tot1 = e_sub.sum()
    print(f"Energy retained after floor subtraction: {tot1/tot0*100:.1f}%\n")

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    cum = np.cumsum(e_sub.ravel()[order]) / tot1

    print("Encircled energy AFTER floor subtraction:")
    for frac in (0.90, 0.95, 0.99, 0.995, 0.999):
        idx = np.searchsorted(cum, frac)
        rr = r_sorted[min(idx, len(r_sorted) - 1)]
        print(f"  {frac*100:5.1f}% within r = {rr:6.1f} px (crop {2*int(np.ceil(rr))}x{2*int(np.ceil(rr))})")

    print("\nMarginal energy per 20-px annulus AFTER subtraction:")
    edges = np.arange(0, 700, 20)
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        ring = e_sub[m].sum() / tot1
        bar = '#' * int(min(50, ring * 500))
        print(f"r=[{edges[i]:4d},{edges[i+1]:4d}): {ring*100:6.3f}%  {bar}")

    if args.out:
        if psf.ndim == 2:
            out = np.clip(psf - floor, 0, None)
            out = out / out.sum()
        else:
            # subtract per-channel floor estimated the same way
            out = psf.copy()
            for c in range(psf.shape[-1]):
                ec = psf[..., c]
                fc = np.median(ec[far])
                out[..., c] = np.clip(ec - fc, 0, None)
                s = out[..., c].sum()
                if s > 0:
                    out[..., c] /= s
        np.save(args.out, out.astype(np.float32))
        print(f"\nSaved floor-subtracted PSF -> {args.out}")


if __name__ == '__main__':
    main()