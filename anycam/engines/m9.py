"""Leica M9 (firmware 1.216) engine: dual-DCP bridge into M9 raw, bf547 AWB, BF561 5x5 colour kernel and B&W.

B&W (the menu's monochrome) is the 11th matrix pair, the Rec. 709 luminance row applied straight to the white-balanced camera
RGB, with its own five tone curves; those curves are laid out for one stop more signal than the colour ones, so the B&W index
is floor(value / 32768) (the 18 % grey then leaves at sRGB 118 - 127, as the colour look's 112 - 124).

Exposure is bridged to the M9 standard rendering at BE_M9 = -0.5: k = 2^(EV + BE_src - BE_M9).
"""
import struct
from pathlib import Path

import numpy as np
import torch

from ..core.bridge import DualDcpBridge
from ..core.color import XYZ_D65, cct_duv
from ..core.dcp import DcpProfile
from ..core.gpu import as_t, map_pixels, resolve_device
from ..core.output import encode_from_srgb_t
from ..core.raw import RawFrame
from .m9_awb import PRESET_NAMES, REF_REFERENCE_BODY, Awb, sample_windows

DEFAULT_M9_LOOK = Path(__file__).resolve().parent.parent.parent / "assets" / "m9" / "leica_m9.bin"
BE_M9 = -0.5

# Adobe Standard ColorMatrix1/2 of the M9 (M9 Digital Camera Adobe Standard.dcp)
M9_CM1 = np.array([[0.6433, -0.2244, 0.0626], [-0.2873, 0.9694, 0.3728], [-0.0204, 0.1456, 0.8029]], dtype=np.float64)
M9_CM2 = np.array([[0.6687, -0.1751, -0.0291], [-0.3556, 1.1373, 0.2492], [-0.0548, 0.2204, 0.7146]], dtype=np.float64)


def decode_m9_bin(path: Path = DEFAULT_M9_LOOK) -> dict:
    """Decode the Leica M9 firmware 1.216 kind-10 look (Adobe FM/HSM + BF561 records + 5 curves + B&W matrix + 5 B&W curves)."""
    b = Path(path).read_bytes()
    if b[:8] != b"MCOLLOOK":
        raise ValueError(f"Invalid look magic in {path}")
    ver, kind = struct.unpack("<II", b[8:16])
    if kind != 10:
        raise ValueError(f"Expected kind 10 in {path}, got version={ver}, kind={kind}")
    at = 16
    scale = struct.unpack_from("<f", b, at)[0]; at += 4
    fm = np.frombuffer(b, "<f4", 18, at).astype(np.float64).reshape(2, 3, 3); at += 72
    records = np.frombuffer(b, "<i4", 90, at).reshape(5, 2, 3, 3).copy(); at += 360
    curves = (np.frombuffer(b, np.uint8, 5 * 2048, at).reshape(5, 2048).astype(np.float32) / 255.0).copy(); at += 5 * 2048
    mono = np.frombuffer(b, "<i4", 9, at).reshape(3, 3).copy(); at += 36
    mono_curves = (np.frombuffer(b, np.uint8, 5 * 2048, at).reshape(5, 2048).astype(np.float32) / 255.0).copy(); at += 5 * 2048
    hd, sd, t1, t2 = struct.unpack_from("<IIff", b, at); at += 16
    n = hd * sd * 3
    tb = [np.frombuffer(b, "<f4", n, at + 4 * n * i).astype(np.float64).reshape(hd, sd, 3).copy() for i in range(2)]
    if at + 8 * n != len(b):
        raise ValueError(f"Corrupt look size in {path}")
    profile = DcpProfile(name="Leica M9 Adobe Standard", cm=[M9_CM1, M9_CM2], fm=[fm[0], fm[1]], hsm=tb, t1=t1, t2=t2)
    return {"scale": float(scale), "profile": profile, "records": records, "curves": curves, "mono": mono, "mono_curves": mono_curves}


def exposure(frame: RawFrame, src: DcpProfile, ev: float) -> tuple[float, int]:
    """(k_to_m9, metered BV x 256) for the frame."""
    raw_base_ev = frame.raw_base_ev(raf_input_scale=True)
    src_base_ev = raw_base_ev + src.baseline_ev
    k = float(2.0 ** (ev + src_base_ev - BE_M9))
    bv = round((frame.ev100 + raw_base_ev - 5.0) * 256) if frame.ev100 is not None else 1500
    return k, bv


def _sample_patches(cam_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 551 sparse 4x4-scaled AWB windows (R/G window and B left-shifted window) in sensor coordinates."""
    H, W, _ = cam_rgb.shape
    ys, xs, xbs, wy, wx = sample_windows(H, W)
    rg = np.stack([cam_rgb[y0:y0 + wy, x0:x0 + wx] for y0 in ys for x0 in xs], axis=0)
    bp = np.stack([cam_rgb[y0:y0 + wy, xb0:xb0 + wx] for y0 in ys for xb0 in xbs], axis=0)
    return rg, bp


def solve_wb(cam_rgb: np.ndarray, src: DcpProfile, src_neutral: np.ndarray, k_to_m9: float, bv: int,
             wb_spec: str, look: dict, awb: Awb, *, ref=REF_REFERENCE_BODY, iso: float | None = None,
             use_hsm: bool = True, device: str = "auto") -> dict:
    """M9 raw gain and common scene white for auto / camera / preset / Kelvin WB."""
    dev = resolve_device(device)
    m9 = look["profile"]
    spec = wb_spec.strip().lower()

    if spec == "camera":
        white, _ = src.white_of_neutral(src_neutral)
        n_m9, _ = m9.neutral_of_white(white)
        g = 1.0 / n_m9
        xy = tuple(int(round(v)) for v in awb.plane_of_gain(g, ref))
        cct, duv = cct_duv(white)
        return dict(mode="camera", white=white, gain=g, xy=xy, cct=cct, duv=duv, rounds=1)

    if spec in PRESET_NAMES:
        xy = awb.preset(spec)
        g = awb.gain_at(*xy, ref)
        white = m9.white_of_neutral(1.0 / g)[0]
        cct, duv = cct_duv(white)
        return dict(mode=spec, white=white, gain=g, xy=xy, cct=cct, duv=duv, rounds=1)

    if spec.endswith("k") or spec.replace(".", "", 1).isdigit():
        k_val = float(spec[:-1] if spec.endswith("k") else spec)
        xy = awb.kelvin(k_val)
        g = awb.gain_at(*xy, ref)
        white = m9.white_of_neutral(1.0 / g)[0]
        cct, duv = cct_duv(white)
        return dict(mode=f"{int(round(k_val))}K", white=white, gain=g, xy=xy, cct=cct, duv=duv, rounds=1)

    if spec != "auto":
        raise ValueError(f"Unsupported --wb '{wb_spec}'. Use auto, camera, {', '.join(PRESET_NAMES)}, or <Kelvin>K.")

    # bf547 AWB on the sparse windows, iterated from D65 to the dual-DCP fixed point
    rg_patches, bp_patches = _sample_patches(cam_rgb)
    with torch.no_grad():
        n_blk, wy, wx, _ = rg_patches.shape
        both = np.concatenate([rg_patches.reshape(-1, 3), bp_patches.reshape(-1, 3)], axis=0)
        both_t = torch.from_numpy(both).to(device=dev, dtype=torch.float32)
        white = XYZ_D65.copy()
        history = []
        g = np.ones(3, dtype=np.float64)
        xy = (276, 236)
        for _ in range(8):
            bridge = DualDcpBridge(src, m9, white, k_to_m9, use_hsm, dev, inverse="iterate", rounds=24)
            m9_both = bridge(both_t).reshape(2 * n_blk, wy * wx, 3).mean(dim=1).cpu().numpy() * 16383.0
            m9_rg, m9_bp = m9_both[:n_blk], m9_both[n_blk:]
            blks = np.zeros((20, 30, 3), dtype=np.float64)
            blks[1:, 1:, 0] = m9_rg[:, 0].reshape(19, 29)
            blks[1:, 1:, 1] = m9_rg[:, 1].reshape(19, 29)
            blks[1:, 1:, 2] = m9_bp[:, 2].reshape(19, 29)
            g, xy = awb.run(blks.reshape(-1, 3), bv, ref, iso=iso)
            if xy in history:
                break
            history.append(xy)
            white = m9.white_of_neutral(1.0 / g)[0]

    cct, duv = cct_duv(white)
    return dict(mode="auto", white=white, gain=g, xy=xy, cct=cct, duv=duv, rounds=len(history) + 1)


def m9_kernel_t(m9raw_wb: torch.Tensor, records_2x3x3: torch.Tensor, curve_2048: torch.Tensor, index_div: float = 65536.0) -> torch.Tensor:
    """BF561 ASMExecuteColorMatrix: white-balanced M9 raw (N, 3) -> sRGB-encoded (N, 3). B&W: both matrices the luminance
    matrix and index_div 32768."""
    px = torch.floor(torch.clamp(m9raw_wb, 0.0, 1.0) * 16383.0)
    first = (px[:, 0] >= px[:, 1]).unsqueeze(-1)
    val = torch.where(first, px @ records_2x3x3[0].T, px @ records_2x3x3[1].T)
    idx = torch.floor(val / index_div).long().clamp(0, 2047)
    return curve_2048[idx]


def render(cam_rgb: np.ndarray, src: DcpProfile, look: dict, wb_state: dict, k_to_m9: float, *,
           contrast: int = 0, saturation: int = 0, mono: bool = False, use_hsm: bool = True, out_space: str = "srgb",
           device: str = "auto", chunk_pixels: int = 8_000_000) -> np.ndarray:
    """Sensor-oriented camera RGB -> dual-DCP -> M9 raw x AWB gain -> BF561 kernel -> encoded output."""
    dev = resolve_device(device)
    bridge = DualDcpBridge(src, look["profile"], wb_state["white"], k_to_m9, use_hsm, dev)
    g_t = as_t(wb_state["gain"], dev)
    step = int(np.clip(round(contrast), -2, 2)) + 2
    if mono:
        rec_t = as_t(np.stack([look["mono"], look["mono"]]), dev)
        curve_t, div = as_t(look["mono_curves"][step], dev), 32768.0
    else:
        rec_t = as_t(look["records"][int(np.clip(round(saturation), -2, 2)) + 2], dev)
        curve_t, div = as_t(look["curves"][step], dev), 65536.0
    fn = lambda c: encode_from_srgb_t(m9_kernel_t(bridge(c) * g_t, rec_t, curve_t, div), out_space, dev)
    return map_pixels(cam_rgb, fn, dev, chunk_pixels)
