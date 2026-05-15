"""
Evaluates real-world deblurring on RealBlur-J, RealBlur-R, and GoPro test
sets in three modes:

  --mode vanilla_restormer  : Standard Restormer + GoPro-pretrained weights.
  --mode conda_backbone      : ConDA-Restormer backbone-only (adapter_id=-1).
                               This is the V1-trained backbone in isolation.
  --mode conda_router        : ConDA-Restormer with prototype-based routing
                               across V1 (backbone) + V2..V6 adapters.

Expected dataset layout (--data_dir points to one of these):
    <data_dir>/
        input_crops/*.png  (blurred images)
        target_crops/*.png  (sharp gt images)
"""

import os
import sys
import json
import argparse
from glob import glob
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["vanilla_restormer", "conda_backbone", "conda_router"])
    p.add_argument("--data_dir", required=True,
                   help="dataset root containing input/ and target/ subdirs")
    p.add_argument("--weights", required=True, help="path to model checkpoint")
    p.add_argument("--yaml_file", default=None,
                   help="(ConDA modes) path to ConDA model YAML")
    p.add_argument("--prototypes", default=None,
                   help="(conda_router) path to prototype bank .pth file")
    p.add_argument("--max_images", type=int, default=None,
                   help="evaluate only first N pairs (sanity check)")
    p.add_argument("--save_outputs", default=None,
                   help="(optional) directory to dump restored PNGs into")
    p.add_argument("--results_json", default=None,
                   help="dump per-image and aggregate results to this JSON file")
    return p.parse_args()


def collect_pairs(data_dir):
    input_dir = os.path.join(data_dir, "input_crops")
    target_dir = os.path.join(data_dir, "target_crops")
    blur_files = natsorted(glob(os.path.join(input_dir, "*.png")))
    pairs = [(b, os.path.join(target_dir, os.path.basename(b))) for b in blur_files]
    return pairs


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()


def to_numpy(t):
    return t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def pad_to_factor(t):
    factor = 8  # padding factor for Restormer
    _, _, h, w = t.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, h, w


def compute_psnr_ssim(pred, gt):
    p = (pred * 255.0).clip(0, 255).astype(np.uint8)
    g = (gt * 255.0).clip(0, 255).astype(np.uint8)
    psnr = float(psnr_fn(g, p, data_range=255))
    ssim = float(ssim_fn(g, p, channel_axis=2, data_range=255))
    return psnr, ssim


def maybe_save_output(out_np, basename, save_dir):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    out_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, basename), out_bgr)


def build_vanilla_restormer(weights_path):  # standard Restormer with GoPro motion deblurring config.
    from basicsr.models.archs.restormer_arch import Restormer

    model = Restormer()
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("params", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def build_conda_restormer(yaml_file, weights_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_domain_incremental import build_model

    class _Args:
        pass

    a = _Args()
    a.yaml_file = yaml_file
    a.weights = weights_path

    model = build_model(a)
    model.cuda().eval()
    return model


class BottleneckExtractor:
    """
    L2-normalized GAP'd bottleneck features using adapter_id=-1
    so embeddings exist in the shared backbone-only feature space.
    """
    def __init__(self, model):
        self.model = model
        self._features = {}
        self._handle = model.latent[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._features["bottleneck"] = output

    def extract_embedding(self, img_tensor):
        with torch.no_grad():
            _ = self.model(img_tensor, adapter_id=-1)  # backbone only

        bottleneck = self._features["bottleneck"]
        embedding = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)  # [1, C]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def close(self):
        self._handle.remove()


def route(embedding, prototypes):
    sims = {}
    e = embedding.squeeze(0)
    for k, p in prototypes.items():
        sims[k] = float(torch.dot(e, p.to(e.device)))
    pred = max(sims, key=sims.get)
    return pred, sims


def key_to_adapter_id(k):
    if str(k).lower().startswith("d"):
        n = int(str(k).lstrip("Dd").split("_")[0].strip())
    else:
        n = int(str(k).lstrip("Vv").strip())
    return n - 2


def _run_image(model, blur, mode, adapter_id):
    x, h, w = pad_to_factor(to_tensor(blur))
    with torch.no_grad():
        y = model(x) if mode == "vanilla_restormer" else model(x, adapter_id=adapter_id)
    return to_numpy(y[..., :h, :w])


def infer(model, pairs, args, get_adapter_id=None, extractor=None, prototypes=None):
    """Unified inference loop. get_adapter_id is a callable producing the
    adapter_id for the current image (None for vanilla_restormer)."""
    psnrs, ssims, per_image = [], [], []
    routing = Counter() if extractor is not None else None

    for blur_path, gt_path in tqdm(pairs, desc=args.mode):
        blur = load_image(blur_path)
        gt = load_image(gt_path)

        record = {"file": os.path.basename(blur_path)}

        if extractor is not None:
            x_pad, _, _ = pad_to_factor(to_tensor(blur))
            emb = extractor.extract_embedding(x_pad)
            pred_key, sims = route(emb, prototypes)
            adapter_id = key_to_adapter_id(pred_key)
            routing[str(pred_key)] += 1
            record.update({
                "predicted_domain": str(pred_key),
                "adapter_id": adapter_id,
                "similarities": {str(k): v for k, v in sims.items()},
            })
        else:
            adapter_id = get_adapter_id() if get_adapter_id else None

        out = _run_image(model, blur, args.mode, adapter_id)
        p, s = compute_psnr_ssim(out, gt)
        psnrs.append(p)
        ssims.append(s)
        record["psnr"] = p
        record["ssim"] = s
        per_image.append(record)
        maybe_save_output(out, os.path.basename(blur_path), args.save_outputs)

    return psnrs, ssims, per_image, routing


def main():
    args = parse_args()

    pairs = collect_pairs(args.data_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]
    print(f"Found {len(pairs)} pairs in {args.data_dir}")

    extractor, prototypes = None, None

    if args.mode == "vanilla_restormer":
        model = build_vanilla_restormer(args.weights)
        get_adapter_id = None
    elif args.mode == "conda_backbone":
        assert args.yaml_file, "--yaml_file required for conda_backbone"
        model = build_conda_restormer(args.yaml_file, args.weights)
        get_adapter_id = lambda: -1
    elif args.mode == "conda_router":
        assert args.yaml_file, "--yaml_file required for conda_router"
        assert args.prototypes, "--prototypes required for conda_router"
        model = build_conda_restormer(args.yaml_file, args.weights)
        proto_data = torch.load(args.prototypes, map_location="cpu")
        prototypes = proto_data["prototypes"]
        print(f"Loaded prototypes for keys: {list(prototypes.keys())}")
        extractor = BottleneckExtractor(model)
        get_adapter_id = None

    psnrs, ssims, per_image, routing = infer(model, pairs, args, 
                                             get_adapter_id=get_adapter_id, 
                                             extractor=extractor, prototypes=prototypes)

    if extractor is not None:
        extractor.close()

    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}")
    print(f"N images: {len(psnrs)}")
    print(f"{'-'*60}")
    print(f"Mean PSNR: {np.mean(psnrs):7.3f} dB, Mean SSIM: {np.mean(ssims):.4f}")
    print(f"Std PSNR: {np.std(psnrs):7.3f} dB, Std SSIM: {np.std(ssims):.4f}")

    if routing is not None:
        total = sum(routing.values())
        print(f"\nRouting distribution over {total} images:")
        for k in natsorted(routing.keys()):
            n = routing[k]
            print(f"domain {k}: {n:5d} ({100.0 * n / total:5.1f}%)")

    if args.results_json:
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump({
                "mode": args.mode,
                "data_dir": args.data_dir,
                "weights": args.weights,
                "n_images": len(psnrs),
                "mean_psnr": float(np.mean(psnrs)),
                "mean_ssim": float(np.mean(ssims)),
                "median_psnr": float(np.median(psnrs)),
                "median_ssim": float(np.median(ssims)),
                "std_psnr": float(np.std(psnrs)),
                "routing": dict(routing) if routing else None,
                "per_image": per_image,
            }, f, indent=2)
        print(f"\nResults JSON written to: {args.results_json}")


if __name__ == "__main__":
    main()
