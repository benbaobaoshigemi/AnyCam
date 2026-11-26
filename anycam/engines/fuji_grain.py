"""PyTorch GPU-accelerated X100VI Film Grain Effect (RTX 4060 CUDA default, CPU fallback).

Matches the calibrated 8-fold symmetric 9x9 / 7x7 / 5x5 + pixelwise tanh + PWL envelope
model, with diagonal-anchored 48 um aperture invariance across arbitrary resolutions.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from ..core.gpu import resolve_device

X100VI_REF_DIAG_PX = float(math.hypot(7728, 5152))
FILM_35MM_DIAG_UM = 43270.0


def _build_sym_kernel(vec, r_half: int) -> np.ndarray:
    n = 2 * r_half + 1
    k = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for dy in range(r_half + 1):
        for dx in range(dy, r_half + 1):
            v = vec[idx]
            idx += 1
            for sy in (-1, 1):
                for sx in (-1, 1):
                    k[r_half + sy * dy, r_half + sx * dx] = v
                    k[r_half + sy * dx, r_half + sx * dy] = v
    return (k / (np.sqrt(np.sum(k ** 2)) + 1e-12)).astype(np.float32)


_SYM_PRESETS = {
    (False, "weak",   "small"): (4, [1.0, 0.33116, 0.14895, -0.00984, -0.01873, 0.24366, 0.08451, -0.02435, -0.01773, 0.0236, -0.02192, -0.01073, -0.01738, -0.00815, -0.00521], 0.0000),
    (False, "strong", "small"): (4, [1.0, 0.40531, 0.15863, -0.02756, -0.02305, 0.27077, 0.08281, -0.0401,  -0.02157, 0.01405, -0.02873, -0.01325, -0.02026, -0.00774, -0.00389], 0.0000),
    (False, "weak",   "large"): (3, [1.0, 0.04991, -0.01990, -0.00460, -0.04136, -0.02545, -0.00465, -0.00220, -0.00187, -0.00176], 0.6532),
    (False, "strong", "large"): (3, [1.0, 0.05841, -0.02504, -0.00040, -0.05131, -0.03431, -0.00312, -0.00159, -0.00239, -0.00042], 0.6942),
    (True,  "weak",   "small"): (4, [1.0, 0.33116, 0.14895, -0.00984, -0.01873, 0.24366, 0.08451, -0.02435, -0.01773, 0.0236, -0.02192, -0.01073, -0.01738, -0.00815, -0.00521], 0.0000),
    (True,  "strong", "small"): (4, [1.0, 0.40531, 0.15863, -0.02756, -0.02305, 0.27077, 0.08281, -0.0401,  -0.02157, 0.01405, -0.02873, -0.01325, -0.02026, -0.00774, -0.00389], 0.0000),
    (True,  "weak",   "large"): (2, [1.0, 0.01164, -0.01969, -0.05149, -0.011, 0.01129], 0.9633),
    (True,  "strong", "large"): (2, [1.0, 0.01164, -0.01969, -0.05149, -0.011, 0.01129], 0.9633),
}
KERNEL_PRESETS = {k: (_build_sym_kernel(vec, r), r, sat) for k, (r, vec, sat) in _SYM_PRESETS.items()}

SIGMA_PLAT_U8 = {
    (False, "weak",   "small"): 3.322,
    (False, "strong", "small"): 7.048,
    (False, "weak",   "large"): 7.676,
    (False, "strong", "large"): 15.860,
    (True,  "weak",   "small"): 3.463,
    (True,  "strong", "small"): 7.347,
    (True,  "weak",   "large"): 8.001,
    (True,  "strong", "large"): 16.532,
}

_COLOR_LUT_Y = np.array([0.000, 0.0375, 0.0754, 0.1315, 0.2251, 0.3419, 0.7315, 0.8492, 0.9194, 1.000], dtype=np.float32)
_COLOR_LUT_G = np.array([0.350, 0.692,  0.865,  0.905,  0.972,  1.000,  1.000,  0.941,  0.883,  0.810], dtype=np.float32)

_ACROS_LUT_Y = np.array([0.000, 0.0375, 0.0754, 0.1315, 0.2251, 0.3419, 0.4582, 0.5957, 0.6597, 0.7315, 0.8492, 0.9194, 1.000], dtype=np.float32)
_ACROS_LUT_G = np.array([0.350, 0.678,  0.870,  0.931,  0.971,  1.000,  1.000,  0.985,  0.963,  0.926,  0.797,  0.710,  0.610], dtype=np.float32)


def _conv2d_circular(field: torch.Tensor, ker: torch.Tensor, pad: int) -> torch.Tensor:
    x = F.pad(field, (pad, pad, pad, pad), mode="circular")
    return F.conv2d(x, ker)


def _aperture_48um_std_t(field: torch.Tensor, px_um: float) -> float:
    sigma_px = (48.0 / 4.0) / px_um
    f = field
    if sigma_px < 0.8:
        up = int(math.ceil(1.0 / max(sigma_px, 0.1)))
        f = F.interpolate(f, scale_factor=up, mode="bilinear", align_corners=False)
        sigma_px = sigma_px * up
    r = int(math.ceil(3.0 * sigma_px))
    ax = torch.arange(-r, r + 1, device=f.device, dtype=torch.float32)
    g1 = torch.exp(-0.5 * (ax / sigma_px) ** 2)
    g1 = g1 / g1.sum()
    kv = g1.view(1, 1, -1, 1)
    kh = g1.view(1, 1, 1, -1)
    tmp = F.conv2d(F.pad(f, (0, 0, r, r), mode="circular"), kv)
    ap = F.conv2d(F.pad(tmp, (r, r, 0, 0), mode="circular"), kh)
    return float(ap.std().item())


def apply_grain_gpu(
    rgb_img: np.ndarray,
    strength: str = "off",
    size: str = "small",
    is_acros: bool = False,
    seed: int = 0,
    resolution_invariant: bool = True,
    device: str = "auto",
) -> np.ndarray:
    """Apply X100VI film grain on (H, W, 3) encoded [0, 1] image using PyTorch GPU/CPU."""
    if strength == "off":
        return np.clip(rgb_img, 0.0, 1.0)
    dev = resolve_device(device)
    H, W, _ = rgb_img.shape
    ker_np, r_half, sat_a = KERNEL_PRESETS[(bool(is_acros), strength, size)]
    sigma_01 = SIGMA_PLAT_U8[(bool(is_acros), strength, size)] / 255.0

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    with torch.no_grad():
        ker_t = torch.from_numpy(ker_np).to(dev).view(1, 1, 2 * r_half + 1, 2 * r_half + 1)

        def _native_sample(hh: int, ww: int) -> torch.Tensor:
            u = torch.randn((1, 1, hh, ww), generator=gen, device=dev, dtype=torch.float32)
            out = _conv2d_circular(u, ker_t, r_half)
            out = out / (out.std() + 1e-12)
            if sat_a > 0.02:
                out = torch.tanh(sat_a * out)
                out = out / (out.std() + 1e-12)
            return out

        ref_px_um = FILM_35MM_DIAG_UM / X100VI_REF_DIAG_PX
        tgt_px_um = FILM_35MM_DIAG_UM / float(math.hypot(H, W))
        scale = tgt_px_um / ref_px_um

        if not resolution_invariant or abs(scale - 1.0) < 0.02:
            field = _native_sample(H, W)[0, 0]
        else:
            h_ref = max(32, int(round(H * scale)))
            w_ref = max(32, int(round(W * scale)))
            n_ref = _native_sample(h_ref, w_ref)
            target_ap = _aperture_48um_std_t(n_ref, ref_px_um)
            if scale < 1.0:
                n_tgt = F.interpolate(n_ref, size=(H, W), mode="bicubic", align_corners=False)
            else:
                sig_lp = 0.55 * scale
                rlp = int(math.ceil(3.0 * sig_lp))
                ax = torch.arange(-rlp, rlp + 1, device=dev, dtype=torch.float32)
                g1 = torch.exp(-0.5 * (ax / sig_lp) ** 2)
                g1 = g1 / g1.sum()
                tmp = F.conv2d(F.pad(n_ref, (0, 0, rlp, rlp), mode="circular"), g1.view(1, 1, -1, 1))
                n_lp = F.conv2d(F.pad(tmp, (rlp, rlp, 0, 0), mode="circular"), g1.view(1, 1, 1, -1))
                n_tgt = F.interpolate(n_lp, size=(H, W), mode="bilinear", align_corners=False)
            cur_ap = _aperture_48um_std_t(n_tgt, tgt_px_um)
            if cur_ap > 1e-12:
                n_tgt = n_tgt * (target_ap / cur_ap)
            field = n_tgt[0, 0]

        img_t = torch.from_numpy(rgb_img).to(device=dev, dtype=torch.float32)
        y = torch.clamp(0.299 * img_t[..., 0] + 0.587 * img_t[..., 1] + 0.114 * img_t[..., 2], 0.0, 1.0)
        xp = torch.from_numpy(_ACROS_LUT_Y if is_acros else _COLOR_LUT_Y).to(dev)
        fp = torch.from_numpy(_ACROS_LUT_G if is_acros else _COLOR_LUT_G).to(dev)
        idx = torch.clamp(torch.searchsorted(xp, y, side="right") - 1, 0, xp.shape[0] - 2)
        t = torch.clamp((y - xp[idx]) / (xp[idx + 1] - xp[idx]), 0.0, 1.0)
        env = fp[idx] * (1.0 - t) + fp[idx + 1] * t

        delta = field * sigma_01 * env
        res = torch.clamp(img_t + delta.unsqueeze(-1), 0.0, 1.0)
        return res.cpu().numpy()
