"""Noise-adaptive RAW denoise on linear camera RGB (GPU).

Runs on the demosaiced linear frame before any engine, so every engine and the M9 AWB see the
same cleaned sensor data. The noise level is measured from the frame itself: neighbour
differences are binned by brightness and a signal-dependent model sigma_d^2 = a * x + b is fitted
with a robust (MAD) estimator, so the strength follows ISO and exposure without per-camera tables.

  luma   Y = (R + 2G + B) / 4      noise-adaptive 5x5 bilateral, range = k * sigma_d(Y)
  chroma R - G, B - G             guided filter on the denoised luma, eps = (k * sigma_d(Y))^2;
                                  where the removed residual has a local mean well above the
                                  chroma noise floor (a colour edge or detail with no luma edge),
                                  the original chroma is kept

Chroma noise is removed far more strongly than luma noise, which keeps texture and edges while
taking out the colour blotches that the colour engines would otherwise amplify.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

DENOISE_LEVELS = ("off", "low", "medium", "high")

# (luma range k, luma blend, chroma radius at 24 MP, chroma eps k, chroma structure threshold)
_PRESETS = {
    "low": (1.0, 0.5, 4, 2.0, 2.0),
    "medium": (1.5, 0.8, 8, 3.0, 3.0),
    "high": (2.0, 1.0, 12, 4.0, 4.0),
}


def fit_noise_model(x: torch.Tensor, level: torch.Tensor | None = None, n_bins: int = 24,
                    max_samples: int = 4_000_000) -> tuple[float, float]:
    """Fit sigma_d^2 = a * level + b for one (H, W) channel from horizontal neighbour differences.

    level defaults to the channel itself; chroma channels pass the luma as their brightness level.
    """
    lv = x if level is None else level
    d = (x[:, 1:] - x[:, :-1]).reshape(-1)
    m = (0.5 * (lv[:, 1:] + lv[:, :-1])).reshape(-1).contiguous()
    if d.numel() > max_samples:
        step = d.numel() // max_samples
        d, m = d[::step], m[::step].contiguous()
    lo, hi = 1e-3, 0.9
    edges = torch.logspace(math.log10(lo), math.log10(hi), n_bins + 1, device=x.device)
    idx = torch.bucketize(m, edges) - 1
    levels, var = [], []
    for k in range(n_bins):
        sel = d[idx == k]
        if sel.numel() < 2000:
            continue
        mad = sel.abs().median()
        levels.append(float(m[idx == k].mean()))
        var.append(float((1.4826 * mad) ** 2))
    if len(levels) < 2:
        s = float((1.4826 * d.abs().median()) ** 2)
        return 0.0, s
    L = np.array(levels)
    V = np.array(var)
    A = np.stack([L, np.ones_like(L)], 1)
    w = 1.0 / np.maximum(V, 1e-12)            # relative-error weighting
    a, b = np.linalg.lstsq(A * w[:, None], V * w, rcond=None)[0]
    if a < 0:
        a, b = 0.0, float(np.median(V))
    if b < 0:
        b = 0.0
        a = float(np.sum(L * V * w * w) / np.sum(L * L * w * w))
    return float(a), float(b)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Separable mean filter with reflect padding on (1, C, H, W)."""
    k = 2 * r + 1
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.avg_pool2d(x, (1, k), stride=1)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.avg_pool2d(x, (k, 1), stride=1)


def _bilateral(y: torch.Tensor, sigma_r: torch.Tensor, radius: int = 2, sigma_s: float = 1.5) -> torch.Tensor:
    """Bilateral filter on (H, W) with a per-pixel range sigma."""
    H, W = y.shape
    yp = F.pad(y[None, None], (radius, radius, radius, radius), mode="reflect")[0, 0]
    inv2r = 1.0 / (2.0 * sigma_r.clamp_min(1e-7) ** 2)
    acc = torch.zeros_like(y)
    wsum = torch.zeros_like(y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = yp[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            w = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s)) * torch.exp(-((nb - y) ** 2) * inv2r)
            acc += w * nb
            wsum += w
    return acc / wsum


def _guided(guide: torch.Tensor, src: torch.Tensor, r: int, eps: torch.Tensor) -> torch.Tensor:
    """Guided filter: guide (1, 1, H, W), src (1, C, H, W), per-pixel eps (1, 1, H, W)."""
    mi = _box(guide, r)
    mp = _box(src, r)
    cov = _box(guide * src, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


def denoise_t(img: torch.Tensor, level: str) -> torch.Tensor:
    """(H, W, 3) linear camera RGB on the target device -> denoised (H, W, 3)."""
    if level == "off":
        return img
    k_l, blend, r24, k_c, t_c = _PRESETS[level]
    H, W, _ = img.shape
    r_c = max(2, int(round(r24 * math.hypot(H, W) / math.hypot(4000, 6000))))
    with torch.no_grad():
        R, G, B = img.unbind(-1)
        Y = 0.25 * R + 0.5 * G + 0.25 * B
        a, b = fit_noise_model(Y)
        sig = torch.sqrt((a * Y.clamp_min(0.0) + b).clamp_min(1e-12))
        Yd = _bilateral(Y, k_l * sig)
        Yn = Y + blend * (Yd - Y)

        chroma = torch.stack((R - G, B - G))[None]
        guide = Yn[None, None]
        eps = (k_c * k_c) * (a * _box(guide, r_c).clamp_min(0.0) + b)
        c = _guided(guide, chroma, r_c, eps)
        # structure gate: a 5x5 mean of what was removed that stands clear of the chroma noise floor
        # is a colour edge or a small coloured detail (no luma edge), not noise; it is put back
        resid = chroma - c
        bias2 = _box(resid, 2) ** 2
        lvl = _box(guide, 2).clamp_min(0.0)
        noise = torch.cat([(lambda ab: ab[0] * lvl + ab[1])(fit_noise_model(ch, Y)) for ch in chroma[0]], 1)
        keep = ((bias2 - t_c * noise / 9.0) / bias2.clamp_min(1e-12)).clamp(0.0, 1.0)
        c = (c + keep * resid)[0]
        Gn = Yn - 0.25 * (c[0] + c[1])
        return torch.stack((Gn + c[0], Gn, Gn + c[1]), -1)


def denoise_np(img: np.ndarray, level: str, dev: torch.device) -> np.ndarray:
    if level == "off":
        return img
    t = torch.from_numpy(img).to(dev)
    return denoise_t(t, level).cpu().numpy()
