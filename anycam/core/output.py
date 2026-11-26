"""Output encoding (sRGB-encoded engine output -> sRGB / Display P3 / Adobe RGB) and single-image export."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import tifffile

from .color import ADOBE_GAMMA, SRGB_TO_ADOBE, SRGB_TO_P3
from .gpu import as_t

OUT_SPACES = ("srgb", "p3", "adobe")


def srgb_oetf_t(lin: torch.Tensor) -> torch.Tensor:
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * torch.clamp(lin, min=1e-9) ** (1.0 / 2.4) - 0.055)


def encode_from_srgb_t(enc_srgb: torch.Tensor, out_space: str, dev: torch.device) -> torch.Tensor:
    """Re-encode an sRGB-encoded [0, 1] image into the requested output space."""
    enc = torch.clamp(enc_srgb, 0.0, 1.0)
    if out_space == "srgb":
        return enc
    lin_srgb = torch.where(enc <= 0.04045, enc / 12.92, torch.clamp((enc + 0.055) / 1.055, min=1e-9) ** 2.4)
    if out_space == "p3":
        lin_dst = torch.clamp(lin_srgb @ as_t(SRGB_TO_P3, dev).T, 0.0, 1.0)
        return srgb_oetf_t(lin_dst)
    if out_space == "adobe":
        lin_dst = torch.clamp(lin_srgb @ as_t(SRGB_TO_ADOBE, dev).T, 0.0, 1.0)
        return lin_dst ** (1.0 / ADOBE_GAMMA)
    raise ValueError(f"Unsupported out_space '{out_space}'")


def quantize(img: np.ndarray, bits: int) -> np.ndarray:
    """round-half-up of clip(img, 0, 1) * peak in float32 (in-place on one temporary)."""
    peak = np.float32(255.0 if bits == 8 else 65535.0)
    tmp = np.clip(img, 0.0, 1.0, dtype=np.float32)
    tmp *= peak
    tmp += np.float32(0.5)
    return tmp.astype(np.uint8 if bits == 8 else np.uint16)


def save_output(img: np.ndarray, out_path: Path) -> None:
    """Write one .png / .jpg / .tif (16-bit) image."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        tifffile.imwrite(str(out_path), quantize(img, 16), photometric="rgb")
    elif ext in (".jpg", ".jpeg"):
        Image.fromarray(quantize(img, 8)).save(str(out_path), quality=95, subsampling=0)
    elif ext == ".png":
        Image.fromarray(quantize(img, 8)).save(str(out_path), compress_level=3)
    else:
        raise ValueError(f"Unsupported output extension '{ext}' (use .png, .jpg or .tif)")
