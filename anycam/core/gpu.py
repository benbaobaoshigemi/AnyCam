"""Torch device selection and the 2-D Adobe HueSatMap (forward, precomputed inverse, fixed-point inverse)."""
import numpy as np
import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(dev)})"
    return "cpu"


def as_t(arr, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(arr, device=device, dtype=dtype)


def map_pixels(img: np.ndarray, fn, dev: torch.device, chunk_pixels: int = 8_000_000) -> np.ndarray:
    """Run fn on (N, 3) float32 device chunks of an (H, W, 3) image and gather (H, W, 3) float32 on the host."""
    H, W, _ = img.shape
    flat = img.reshape(-1, 3)
    N = flat.shape[0]
    out = np.empty((N, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, chunk_pixels):
            end = min(start + chunk_pixels, N)
            c = torch.from_numpy(flat[start:end]).to(device=dev, dtype=torch.float32)
            out[start:end] = fn(c).cpu().numpy()
    return out.reshape(H, W, 3)


def rgb_to_hsv_t(c: torch.Tensor):
    mx, mn = c.max(-1).values, c.min(-1).values
    d = mx - mn
    r, g, b = c.unbind(-1)
    ds = torch.where(d > 1e-7, d, torch.ones_like(d))
    h = torch.where(mx == r, ((g - b) / ds) % 6.0, torch.where(mx == g, (b - r) / ds + 2.0, (r - g) / ds + 4.0))
    h = torch.where(d > 1e-7, h, torch.zeros_like(h))
    s = torch.where(mx > 1e-7, d / mx.clamp_min(1e-7), torch.zeros_like(mx))
    return h, s, mx


def hsv_to_rgb_t(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = h % 6.0
    c = v * s
    x = c * (1.0 - torch.abs(h % 2.0 - 1.0))
    m = v - c
    i = torch.floor(h).long().clamp(0, 5)
    z = torch.zeros_like(c)
    r = torch.stack([c, x, z, z, x, c], -1).gather(-1, i[..., None])[..., 0]
    g = torch.stack([x, c, c, x, z, z], -1).gather(-1, i[..., None])[..., 0]
    b = torch.stack([z, z, x, c, c, x], -1).gather(-1, i[..., None])[..., 0]
    return torch.stack([r + m, g + m, b + m], -1)


def lookup_hsm_t(h: torch.Tensor, s: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    hd, sd, _ = table.shape
    hs, ss = h * hd / 6.0, s * (sd - 1)
    h0 = torch.floor(hs).long()
    hf = (hs - h0)[..., None]
    h0 = h0 % hd
    h1 = (h0 + 1) % hd
    s0 = torch.floor(ss).long().clamp(0, sd - 2)
    sf = (ss - s0)[..., None]
    return (
        (table[h0, s0] * (1.0 - hf) + table[h1, s0] * hf) * (1.0 - sf)
        + (table[h0, s0 + 1] * (1.0 - hf) + table[h1, s0 + 1] * hf) * sf
    )


def apply_hsm_t(pp: torch.Tensor, table: torch.Tensor, srgb_encoded: bool = False) -> torch.Tensor:
    """Forward 2-D HueSatMap on linear ProPhoto (N, 3)."""
    if srgb_encoded:
        pp = torch.clamp(pp, min=0.0) ** (1.0 / 2.2)
    h, s, v = rgb_to_hsv_t(pp)
    m = lookup_hsm_t(h, s, table)
    out = hsv_to_rgb_t(h + m[..., 0] * 6.0 / 360.0, (s * m[..., 1]).clamp(0.0, 1.0), (v * m[..., 2]).clamp_min(0.0))
    if srgb_encoded:
        out = out ** 2.2
    return out


def build_inv_hsm_t(table: torch.Tensor, hd_inv: int = 360, sd_inv: int = 101, rounds: int = 80) -> torch.Tensor:
    """Precompute a 2-D inverse HueSatMap table (hd_inv, sd_inv, 3) once per frame."""
    hg = torch.arange(hd_inv, device=table.device, dtype=table.dtype) * (6.0 / hd_inv)
    sg = torch.linspace(0.0, 1.0, sd_inv, device=table.device, dtype=table.dtype)
    hh, ss = torch.meshgrid(hg, sg, indexing="ij")
    h1, s1 = hh.reshape(-1), ss.reshape(-1)
    h, s = h1.clone(), s1.clone()
    for _ in range(rounds):
        m = lookup_hsm_t(h, s, table)
        h = (h1 - m[..., 0] * 6.0 / 360.0) % 6.0
        s = (s1 / m[..., 1].clamp_min(1e-4)).clamp(0.0, 1.0)
    m = lookup_hsm_t(h, s, table)
    dh = ((h - h1 + 3.0) % 6.0 - 3.0) * (360.0 / 6.0)
    ms = torch.where(s1 > 1e-6, s / s1.clamp_min(1e-6), 1.0 / m[..., 1].clamp_min(1e-4))
    mv = 1.0 / m[..., 2].clamp_min(1e-4)
    return torch.stack([dh, ms, mv], dim=-1).reshape(hd_inv, sd_inv, 3)


def inverse_hsm_t(pp: torch.Tensor, table: torch.Tensor, rounds: int = 40, inv_table: torch.Tensor | None = None) -> torch.Tensor:
    """Fixed-point inverse HueSatMap; with inv_table: precomputed warm start + 2 polish steps."""
    h1, s1, v1 = rgb_to_hsv_t(pp.clamp_min(0.0))
    if inv_table is not None:
        m0 = lookup_hsm_t(h1, s1, inv_table)
        h = (h1 + m0[..., 0] * 6.0 / 360.0) % 6.0
        s = (s1 * m0[..., 1]).clamp(0.0, 1.0)
        n_iters = 2
    else:
        h, s = h1.clone(), s1.clone()
        n_iters = rounds
    for _ in range(n_iters):
        m = lookup_hsm_t(h, s, table)
        h = (h1 - m[..., 0] * 6.0 / 360.0) % 6.0
        s = (s1 / m[..., 1].clamp_min(1e-4)).clamp(0.0, 1.0)
    m = lookup_hsm_t(h, s, table)
    return hsv_to_rgb_t(h, s, v1 / m[..., 2].clamp_min(1e-4))
