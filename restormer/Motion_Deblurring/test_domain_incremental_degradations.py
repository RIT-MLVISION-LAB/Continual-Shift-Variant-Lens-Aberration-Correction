# ------------------------------------------------------------------------
# Restormer Adapters - Degradation Domain-Incremental Evaluation Script
# Usage:
#   PYTHONPATH=.. python test_domain_incremental_degradations.py \
#       --weights ../experiments/archived_checkpoints/D1_thru_D5_Adapters/net_g_latest.pth \
#       --yaml_file Options/Degradations_D1_thru_D5_Restormer_Adapters.yml \
#       --prototypes degradation_prototypes.pth \
#       --data_root Datasets/val
# ------------------------------------------------------------------------

import os
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import cv2
import yaml

from basicsr.models.archs.restormer_adapters_arch import RestormerAdapters
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


DOMAINS = ["D1_deblur", "D2_denoise", "D3_dehaze", "D4_derain", "D5_lowlight"]

# Maps domain name to the Full_Images validation subdirectory
DOMAIN_TO_VAL_DIR = {
    "D1_deblur":   "ShiftVariant_V1_Full_Images",
    "D2_denoise":  "D2_denoise_Full_Images",
    "D3_dehaze":   "D3_dehaze_Full_Images",
    "D4_derain":   "D4_derain_Full_Images",
    "D5_lowlight": "D5_lowlight_Full_Images",
}

# Maps domain name to adapter_id: D1=-1 (backbone), D2=0, D3=1, D4=2, D5=3
DOMAIN_TO_ADAPTER = {
    "D1_deblur": -1,
    "D2_denoise": 0,
    "D3_dehaze": 1,
    "D4_derain": 2,
    "D5_lowlight": 3,
}


def load_config(yaml_path):
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=Loader)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build_model(args):
    config = load_config(args.yaml_file)
    network_config = config["network_g"].copy()
    network_config.pop("type", None)
    adapter_config = network_config.pop("adapter_config", None)

    model = RestormerAdapters(**network_config, adapter_config=adapter_config)

    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)

    model.prepare_adapter_list_for_loading(state_dict)
    model.load_state_dict(state_dict, strict=False)

    num_committed = len(model.adapter_list)
    print(f"Loaded checkpoint: {args.weights}")
    print(f"Committed adapters: {num_committed}")
    print(f"Available adapter IDs:")
    print("-1 (backbone / D1_deblur)")
    print(f"Committed: {', '.join(str(i) for i in range(num_committed))}")
    print(f"Current: {num_committed}")

    return model


class BottleneckExtractor:
    """
    Hooks the last latent TransformerBlock to capture bottleneck features.
    Used only for domain identification (backbone-only forward pass).
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

    def restore(self, img_tensor, adapter_id):
        with torch.no_grad():
            restored = self.model(img_tensor, adapter_id=adapter_id)
        return restored

    def close(self):
        self._handle.remove()


def match_domain(embedding, prototypes):
    similarities = {}
    emb = embedding.squeeze(0)  # [C]

    for domain, proto in prototypes.items():
        proto = proto.to(emb.device)
        sim = torch.dot(emb, proto).item()
        similarities[domain] = sim

    predicted_domain = max(similarities, key=similarities.get)
    return predicted_domain


def evaluate(args):
    model = build_model(args)
    model.cuda()
    model.eval()

    extractor = BottleneckExtractor(model)

    proto_data = torch.load(args.prototypes, map_location="cpu")
    prototypes = proto_data["prototypes"]
    proto_meta = proto_data.get("metadata", {})

    print(f"\nLoaded prototypes from: {args.prototypes}")
    for domain, meta in sorted(proto_meta.items()):
        print(f"  {domain}: dim={meta['embedding_dim']}, samples={meta['num_samples']}, adapter_id={meta['adapter_id']}")

    domain_list = sorted(prototypes.keys())
    num_domains = len(domain_list)
    print(f"\nDomain-incremental evaluation over {num_domains} domains")

    domains_to_test = []
    for domain in domain_list:
        val_dir_name = DOMAIN_TO_VAL_DIR[domain]
        inp_dir = os.path.join(args.data_root, val_dir_name, "input_crops")
        gt_dir = os.path.join(args.data_root, val_dir_name, "target_crops")
        if os.path.exists(inp_dir):
            domains_to_test.append((domain, inp_dir, gt_dir))
        else:
            print(f"[WARNING] No test data found for {domain} at {inp_dir}, skipping")

    if not domains_to_test:
        print("No test data found. Check --data_root.")
        return

    factor = 8  # padding factor

    results = defaultdict(
        lambda: {
            "psnr_predicted": [],
            "ssim_predicted": [],
            "psnr_oracle": [],
            "ssim_oracle": [],
            "correct": 0,
            "total": 0,
            "predictions": [],
        }
    )

    total_images = 0
    total_correct = 0

    for gt_domain, inp_dir, gt_dir in domains_to_test:
        gt_adapter_id = DOMAIN_TO_ADAPTER[gt_domain]

        files = natsorted(glob(os.path.join(inp_dir, "*.png")))
        if not files:
            print(f"\n{gt_domain}: no images found in {inp_dir}")
            continue

        has_gt = os.path.exists(gt_dir)

        print(f"\nTesting {gt_domain} ({len(files)} images, oracle adapter_id={gt_adapter_id})")

        for file_ in tqdm(files, desc=gt_domain, unit="img"):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(load_img(file_)) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
            _, _, h, w = img_tensor.shape
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            if pad_h > 0 or pad_w > 0:
                img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "reflect")

            # --- Pass 1: backbone-only → domain identification ---
            embedding = extractor.extract_embedding(img_tensor)
            predicted_domain = match_domain(embedding.cpu(), prototypes)
            predicted_adapter_id = DOMAIN_TO_ADAPTER[predicted_domain]

            is_correct = predicted_domain == gt_domain
            total_correct += int(is_correct)
            total_images += 1

            results[gt_domain]["total"] += 1
            results[gt_domain]["correct"] += int(is_correct)
            results[gt_domain]["predictions"].append(predicted_domain)

            # --- Pass 2: predicted adapter → restoration ---
            restored_pred = extractor.restore(img_tensor, adapter_id=predicted_adapter_id)
            restored_pred = torch.clamp(restored_pred[:, :, :h, :w], 0, 1).cpu()

            # --- Oracle adapter → restoration (for comparison) ---
            restored_oracle = extractor.restore(img_tensor, adapter_id=gt_adapter_id)
            restored_oracle = torch.clamp(restored_oracle[:, :, :h, :w], 0, 1).cpu()

            if has_gt:
                gt_path = os.path.join(gt_dir, os.path.basename(file_))
                if os.path.exists(gt_path):
                    gt_img = np.float32(load_img(gt_path)) / 255.0

                    pred_img = restored_pred.squeeze(0).permute(1, 2, 0).numpy()
                    psnr_pred = psnr(gt_img, pred_img, data_range=1.0)
                    ssim_pred = ssim(gt_img, pred_img, data_range=1.0, channel_axis=2)

                    oracle_img = restored_oracle.squeeze(0).permute(1, 2, 0).numpy()
                    psnr_oracle = psnr(gt_img, oracle_img, data_range=1.0)
                    ssim_oracle = ssim(gt_img, oracle_img, data_range=1.0, channel_axis=2)

                    results[gt_domain]["psnr_predicted"].append(psnr_pred)
                    results[gt_domain]["ssim_predicted"].append(ssim_pred)
                    results[gt_domain]["psnr_oracle"].append(psnr_oracle)
                    results[gt_domain]["ssim_oracle"].append(ssim_oracle)

    extractor.close()

    print("\nDEGRADATION DOMAIN-INCREMENTAL EVALUATION RESULTS")
    print("-" * 70)
    overall_acc = 100 * total_correct / total_images if total_images > 0 else 0
    print(f"Domain Identification Accuracy: {total_correct}/{total_images} = {overall_acc:.1f}%")
    print(f"\n{'Domain':<10} {'Accuracy':<12} {'PSNR(pred)':<14} "
          f"{'PSNR(oracle)':<14} {'SSIM(pred)':<14} {'SSIM(oracle)':<14}")
    print("-" * 80)

    all_psnr_pred, all_psnr_oracle = [], []
    all_ssim_pred, all_ssim_oracle = [], []

    for domain in sorted(results.keys()):
        r = results[domain]
        acc = 100 * r["correct"] / r["total"] if r["total"] > 0 else 0

        psnr_p = np.mean(r["psnr_predicted"]) if r["psnr_predicted"] else 0
        psnr_o = np.mean(r["psnr_oracle"]) if r["psnr_oracle"] else 0
        ssim_p = np.mean(r["ssim_predicted"]) if r["ssim_predicted"] else 0
        ssim_o = np.mean(r["ssim_oracle"]) if r["ssim_oracle"] else 0

        all_psnr_pred.extend(r["psnr_predicted"])
        all_psnr_oracle.extend(r["psnr_oracle"])
        all_ssim_pred.extend(r["ssim_predicted"])
        all_ssim_oracle.extend(r["ssim_oracle"])

        print(f"{domain:<10} {acc:>5.1f}%      {psnr_p:>6.2f} dB     "
              f"{psnr_o:>6.2f} dB     {ssim_p:>6.4f}       {ssim_o:>6.4f}")

    print("-" * 80)
    if all_psnr_pred:
        print(f"{'Overall':<10} {overall_acc:>5.1f}%      "
              f"{np.mean(all_psnr_pred):>6.2f} dB     "
              f"{np.mean(all_psnr_oracle):>6.2f} dB     "
              f"{np.mean(all_ssim_pred):>6.4f}       "
              f"{np.mean(all_ssim_oracle):>6.4f}")

    if all_psnr_pred and all_psnr_oracle:
        gap = np.mean(all_psnr_oracle) - np.mean(all_psnr_pred)
        print(f"\nPSNR gap (oracle - predicted): {gap:+.2f} dB")

    n = len(domain_list)
    d_to_idx = {d: i for i, d in enumerate(domain_list)}

    confusion = np.zeros((n, n), dtype=int)
    for gt_d in domain_list:
        if gt_d not in results:
            continue
        for pred_d in results[gt_d]["predictions"]:
            if pred_d in d_to_idx:
                confusion[d_to_idx[gt_d], d_to_idx[pred_d]] += 1

    SHORT = {"D1_deblur": "D1", "D2_denoise": "D2", "D3_dehaze": "D3",
             "D4_derain": "D4", "D5_lowlight": "D5"}

    print(f"\nConfusion Matrix (rows=GT, cols=predicted):")
    header = "       " + "".join(f"  {SHORT[d]:<5}" for d in domain_list)
    print(header)
    for i, gt_d in enumerate(domain_list):
        row = f"  {SHORT[gt_d]:<5} "
        for j in range(n):
            val = confusion[i, j]
            if i == j:
                row += f" [{val:>3}]"
            else:
                row += f"  {val:>3} "
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Degradation domain-incremental evaluation with prototype-based domain identification for RestormerAdapters"
    )
    parser.add_argument(
        "--weights", type=str,
        default="../experiments/archived_checkpoints/Degradations_D1_D2_D3_D4_ft_D5_Adapters/net_g_latest.pth",
        help="Path to RestormerAdapters checkpoint with all adapters",
    )
    parser.add_argument(
        "--yaml_file", type=str,
        default="Options/Degradations_D1_D2_D3_D4_ft_D5_Restormer_Adapters.yml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--prototypes", type=str,
        default="../experiments/archived_checkpoints/prototypes/degradation_prototypes.pth",
        help="Path to degradation prototypes .pth file",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="Datasets/val",
        help="Root directory containing validation subdirectories",
    )

    args = parser.parse_args()
    evaluate(args)
