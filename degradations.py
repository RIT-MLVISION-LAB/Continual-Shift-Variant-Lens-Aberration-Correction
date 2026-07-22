"""
Degradation synthesis functions for cross-task domain-incremental restoration.
All functions operate on float32 images in [0, 1] range, RGB channel order.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


### Gaussian Noise Synthesis

def add_gaussian_noise(img, sigma=None):
    if sigma is None:
        sigma = np.random.uniform(25, 70)
    noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(img + noise, 0.0, 1.0)


### Haze Synthesis (Atmospheric Scattering Model)

_midas_cache = {}

def _get_midas_model():
    """
    Lazy-loads and caches the MiDaS DPT-Hybrid model.
    """
    if "model" not in _midas_cache:
        import torch
        import warnings
 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[degradations] Loading MiDaS DPT-Hybrid on {device}")
 
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
            warnings.filterwarnings("ignore", category=UserWarning, module="timm")
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            model.to(device).eval()
    
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = transforms.dpt_transform
 
        _midas_cache["model"] = model
        _midas_cache["transform"] = transform
        _midas_cache["device"] = device
        print("[degradations] MiDaS loaded")
 
    return _midas_cache["model"], _midas_cache["transform"], _midas_cache["device"]

def _estimate_depth_midas(img):
    """
    Estimates a depth map using MiDaS DPT-Hybrid monocular depth estimation model for scene-aware haze
    """
    import torch
 
    model, transform, device = _get_midas_model()
    h, w = img.shape[:2]
 
    img_uint8 = (img * 255).astype(np.uint8)
    input_batch = transform(img_uint8).to(device)
 
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
 
    depth = prediction.cpu().numpy().astype(np.float32)
 
    # MiDaS outputs inverse depth (higher = closer); invert for scattering
    depth = depth.max() - depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def add_haze(img, beta=None, A=None):
    """
    Synthesizes haze via the atmospheric scattering model:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * d(x)).

    Args:
        img: Clean image.
        depth_map: Precomputed depth map (H, W) in [0, 1].
        beta: Scattering coefficient.
        A: Global atmospheric light in [0, 1].
    """
    try:
        depth_map = _estimate_depth_midas(img)
    except (ImportError, Exception) as e:
        if "model" not in _midas_cache:
            raise RuntimeError(f"[degradations] Pre-trained MiDaS model unavailable ({e})")

    if beta is None:
        beta = np.random.uniform(0.2, 0.5)
    if A is None:
        A = np.random.uniform(0.6, 0.7)
 
    # scaling depth to a physically meaningful range [0.5, 2.0] meters equiv, matching RESIDE-OTS range
    d = depth_map * 1.5 + 0.5
 
    # transmission map
    t = np.exp(-beta * d).astype(np.float32)
    t = np.clip(t, 0.05, 1.0)  # lower-bound to avoid division issues
    t = t[:, :, None]  # (H, W, 1) for broadcasting
 
    hazy = img * t + A * (1.0 - t)
    return np.clip(hazy, 0.0, 1.0).astype(np.float32)


### Rain Streak Synthesis

def _create_rain_layer(h, w, num_streaks=800, streak_length_range=(20, 60), 
                       streak_thickness_range=(1, 3), angle_range=(-15, 15)):
    """
    Generates a single rain streak layer as an alpha map.
    Streaks are near-vertical lines with slight angular variation,
    following the standard parametric rain model.
    """
    rain_layer = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_streaks):
        # random position
        x = np.random.randint(0, w)
        y = np.random.randint(-h // 4, h)

        # streak parameters
        length = np.random.randint(*streak_length_range)
        thickness = np.random.randint(*streak_thickness_range)
        angle = np.random.uniform(*angle_range)  # degrees from vertical

        # streak intensity (brighter streaks are rarer)
        intensity = np.random.uniform(0.4, 1.0)

        # computing endpoints
        angle_rad = np.deg2rad(angle)
        dx = int(length * np.sin(angle_rad))
        dy = int(length * np.cos(angle_rad))

        pt1 = (x, y)
        pt2 = (x + dx, y + dy)

        cv2.line(rain_layer, pt1, pt2, float(intensity), thickness, lineType=cv2.LINE_AA)

    # applying directional motion blur along the streak direction to soften
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[:, kernel_size // 2] = 1.0 / kernel_size
    rain_layer = cv2.filter2D(rain_layer, -1, kernel)

    return rain_layer


def add_rain(img, num_layers=None, streak_density=None, alpha=None, veil=None):
    """
    Synthesizes rain streaks via multi-layer parametric composition with
    atmospheric veiling to model the global visibility reduction caused
    by suspended water droplets scattering light.
    Model:
        1. Streak composite: J_streak = J + alpha * SUM_l(S_l)
        2. Atmospheric veiling: I_rain = (1 - veil) * J_streak + veil * A_rain
    Args:
        img: Clean image.
        num_layers: Number of rain layers (depth planes).
        streak_density: Base number of streaks per layer.
        alpha: Global rain streak intensity.
        veil: Atmospheric veiling factor in [0, 1]. Controls global
              contrast reduction from suspended droplet scattering.
    """
    h, w = img.shape[:2]

    if num_layers is None:
        num_layers = np.random.randint(3, 10)
    if streak_density is None:
        streak_density = np.random.randint(800, 2000)
    if alpha is None:
        alpha = np.random.uniform(0.6, 0.9)
    if veil is None:
        veil = np.random.uniform(0.2, 0.4)

    rain_accumulator = np.zeros((h, w), dtype=np.float32)

    for layer_idx in range(num_layers):
        # farther layers: thinner, shorter, more blurred streaks
        depth_factor = (layer_idx + 1) / num_layers

        layer_streaks = int(streak_density * (1.0 - 0.3 * depth_factor))
        length_range = (int(20 * (1.0 + 0.5 * depth_factor)), int(60 * (1.0 + 0.3 * depth_factor)))
        thickness_range = (1, max(2, int(3 * (1.0 - 0.3 * depth_factor))))

        layer = _create_rain_layer(h, w, layer_streaks, length_range, thickness_range)

        # depth-of-field blur for far layers
        if layer_idx > 0:
            blur_sigma = 0.5 + depth_factor * 1.5
            layer = gaussian_filter(layer, sigma=blur_sigma)

        rain_accumulator += layer * (1.0 - 0.2 * depth_factor)

    # streak composite
    rain_accumulator = np.clip(rain_accumulator, 0, 1)
    rain_rgb = rain_accumulator[:, :, None] * np.array([0.85, 0.85, 0.90], dtype=np.float32)
    streaked = img + alpha * rain_rgb
    streaked = np.clip(streaked, 0.0, 1.0)

    # atmospheric veiling from suspended droplets (global contrast reduction)
    A_rain = 0.7  # gray atmospheric light from water droplet scattering
    rainy = (1.0 - veil) * streaked + veil * A_rain

    return np.clip(rainy, 0.0, 1.0).astype(np.float32)


### Low-Light Synthesis

def add_low_light(img, gamma=None, noise_scale=None):
    """
    Simulates low-light capture via gamma darkening + Poisson-Gaussian noise.
    The degradation model follows the low-light imaging pipeline:
        1. Darken: I_dark = I_clean ^ gamma   (simulates reduced exposure)
        2. Shot noise: I_shot ~ Poisson(I_dark * k) / k
        3. Read noise: I_noisy = I_shot + N(0, sigma_read)
    Args:
        img: Clean image.
        gamma: Darkening gamma (> 1 means darker).
        noise_scale: Controls overall noise magnitude.
    """
    if gamma is None:
        gamma = np.random.uniform(1.3, 2.8)
    if noise_scale is None:
        noise_scale = np.random.uniform(0.01, 0.04)

    # Gamma darkening
    darkened = np.power(np.clip(img, 1e-8, 1.0), gamma).astype(np.float32)

    # Poisson shot noise (signal-dependent)
    k = 1.0 / (noise_scale + 1e-8)  # scale factor to control the shot noise level
    noisy = np.random.poisson(np.clip(darkened * k, 0, None)).astype(np.float32) / k

    # Gaussian read noise (signal-independent)
    sigma_read = noise_scale * 0.5
    read_noise = np.random.randn(*img.shape).astype(np.float32) * sigma_read
    noisy = noisy + read_noise

    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


DEGRADATION_FUNCTIONS = {
    "denoise": add_gaussian_noise,
    "dehaze": add_haze,
    "derain": add_rain,
    "lowlight": add_low_light,
}

# Full domain sequence (D1 is external; D2-D5 are synthesized here)
DOMAIN_ORDER = ["deblur_v1", "denoise", "dehaze", "derain", "lowlight"]
SYNTH_DOMAINS = ["denoise", "dehaze", "derain", "lowlight"]
