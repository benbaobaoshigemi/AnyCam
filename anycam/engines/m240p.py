"""Leica M (Typ 240) / M-P (firmware 2.1.0.0) engine: Film Modes Standard 5x5 / Vivid / Natural and Monochrome.

WB and the dual-DCP hand-off stay inside one calibrator: M240 Adobe Standard CM / FM / HSM for the
scene white, the gain and the bridge (Leica DNG-embedded ColorMatrix is never mixed with Adobe FM/HSM).
--wb auto runs the camera's own IMG-DSP-212 AWB (m240_awb.py) on its 30 x 30 windows of the bridged M240 raw, iterated
from D65 to the dual-DCP fixed point.
Exposure: DNG WhiteLevel 15000 shifted right 2 bits into the 12-bit colorcomp scale (3750), BE_M240 = 0.0.
"""
import struct
from pathlib import Path

import numpy as np
import torch

from ..core.bridge import DualDcpBridge
from ..core.color import XYZ_D65, cct_duv, parse_wb_cct_duv, white_xyz_from_cct_duv
from ..core.dcp import DcpProfile
from ..core.gpu import as_t, map_pixels, resolve_device
from ..core.output import encode_from_srgb_t
from ..core.raw import RawFrame
from .m240_awb import WHITE_LEVEL, Awb, blocks_of_means, sample_windows

DEFAULT_M240P_LOOK = Path(__file__).resolve().parent.parent.parent / "assets" / "m240p" / "leica_m240p.bin"
BE_M240 = 0.0
M240_WHITE_LEVEL = 3750.0
TONE_LEN = 4096
GAMMA_LEN = 4096
LUMISAT_LEN = 256
MODES = ("standard", "vivid", "natural")

# Adobe Standard ColorMatrix1/2 of the M (Typ 240) (LEICA M (Typ 240) Adobe Standard.dcp)
M240_CM1 = np.array([[0.7943, -0.2597, -0.0716], [-0.4639, 1.4357, 0.0141], [-0.0587, 0.2188, 0.7608]], dtype=np.float64)
M240_CM2 = np.array([[0.7199, -0.2140, -0.0712], [-0.4005, 1.3327, 0.0649], [-0.0810, 0.2521, 0.6673]], dtype=np.float64)


def decode_m240p_bin(path: Path = DEFAULT_M240P_LOOK) -> dict:
    """Decode the M240 look (colorcomp, 7 tone_gain, gamma, 7 lumisat) + Adobe Standard FM/HSM."""
    b = Path(path).read_bytes()
    if b[:8] != b"M240LOOK":
        raise ValueError(f"Invalid look magic in {path}")
    ver, kind = struct.unpack("<II", b[8:16])
    if (ver, kind) != (1, 240):
        raise ValueError(f"Unsupported M240 look header version={ver}, kind={kind} in {path}")
    at = 16
    colorcomp = np.frombuffer(b, "<f4", 9, at).astype(np.float64).reshape(3, 3).copy(); at += 36
    tone_gain = (np.frombuffer(b, "<u2", 7 * TONE_LEN, at).reshape(7, TONE_LEN).astype(np.float32) / 2048.0).copy(); at += 7 * TONE_LEN * 2
    gamma = (np.frombuffer(b, np.uint8, GAMMA_LEN, at).astype(np.float32) / 255.0).copy(); at += GAMMA_LEN
    lumisat = (np.frombuffer(b, np.uint8, 7 * LUMISAT_LEN, at).reshape(7, LUMISAT_LEN).astype(np.float32) / 128.0).copy(); at += 7 * LUMISAT_LEN
    fm1 = np.frombuffer(b, "<f4", 9, at).astype(np.float64).reshape(3, 3).copy(); at += 36
    fm2 = np.frombuffer(b, "<f4", 9, at).astype(np.float64).reshape(3, 3).copy(); at += 36
    hd, sd, dcp_t1, dcp_t2 = struct.unpack_from("<IIff", b, at); at += 16
    n = hd * sd * 3
    hs1 = np.frombuffer(b, "<f4", n, at).astype(np.float64).reshape(hd, sd, 3).copy(); at += 4 * n
    hs2 = np.frombuffer(b, "<f4", n, at).astype(np.float64).reshape(hd, sd, 3).copy(); at += 4 * n
    if at != len(b):
        raise ValueError(f"Corrupt look size in {path}")
    adobe_profile = DcpProfile(name="Leica M (Typ 240) Adobe Standard", cm=[M240_CM1, M240_CM2], fm=[fm1, fm2], hsm=[hs1, hs2], t1=dcp_t1, t2=dcp_t2)
    return {"colorcomp": colorcomp, "tone_gain": tone_gain, "gamma": gamma, "lumisat": lumisat, "adobe_profile": adobe_profile}


def exposure(frame: RawFrame, src: DcpProfile, ev: float) -> float:
    src_base_ev = frame.raw_base_ev(raf_input_scale=True) + src.baseline_ev
    return float(2.0 ** (ev + src_base_ev - BE_M240))


def metered_bv(frame: RawFrame) -> int:
    """Metered BV x 256 of the frame (APEX, the source's own raw exposure base)."""
    if frame.ev100 is None:
        return 1500
    return round((frame.ev100 + frame.raw_base_ev(raf_input_scale=True) - 5.0) * 256)


def _auto(cam_rgb: np.ndarray, src: DcpProfile, adobe: DcpProfile, k_to_m240: float, bv: int, awb: Awb,
          use_hsm: bool, device: str) -> dict:
    """The M240 AWB on its 30 x 30 windows of the bridged M240 raw, iterated from D65 to the dual-DCP fixed point."""
    dev = resolve_device(device)
    H, W, _ = cam_rgb.shape
    ys, xs, wy, wx = sample_windows(H, W)
    patches = np.stack([cam_rgb[y0:y0 + wy, x0:x0 + wx] for y0 in ys for x0 in xs], axis=0)
    patches_t = torch.from_numpy(np.ascontiguousarray(patches.reshape(-1, 3))).to(device=dev, dtype=torch.float32)
    white = XYZ_D65.copy()
    history = []
    gains, xy = (256, 256, 256), (0, 0)
    with torch.no_grad():
        for _ in range(8):
            bridge = DualDcpBridge(src, adobe, white, k_to_m240, use_hsm, dev, inverse="iterate", rounds=24)
            means = bridge(patches_t).reshape(900, wy * wx, 3).mean(dim=1).cpu().numpy()
            gains, xy = awb.run(blocks_of_means(means), bv)
            if xy in history or gains[0] <= 0 or gains[2] <= 0:
                break
            history.append(xy)
            white = adobe.white_of_neutral(np.array([256.0 / gains[0], 1.0, 256.0 / gains[2]]))[0]
    return {"white": white, "gains": gains, "xy": xy, "rounds": len(history) + 1}


def solve_wb(src: DcpProfile, src_neutral: np.ndarray, wb_spec: str, look: dict, *, cam_rgb: np.ndarray | None = None,
             k_to_m240: float = 1.0, bv: int = 1500, use_hsm: bool = True, device: str = "auto") -> dict:
    """camera: source Adobe CM -> scene white; CCT[,Duv]: the given white; auto: the M240's own AWB (needs cam_rgb).
    Gain from the M240 Adobe CM for camera / CCT, the firmware's Q8 gains for auto."""
    adobe = look["adobe_profile"]
    if wb_spec.strip().lower() == "auto":
        if cam_rgb is None:
            raise ValueError("--wb auto needs the frame")
        r = _auto(cam_rgb, src, adobe, k_to_m240, bv, Awb(), use_hsm, device)
        cct_scene, duv_scene = cct_duv(r["white"])
        neutral = np.array([256.0 / r["gains"][0], 1.0, 256.0 / r["gains"][2]])
        return {
            "mode": "auto",
            "white_scene": r["white"],
            "cct_scene": float(cct_scene),
            "duv_scene": float(duv_scene),
            "neutral_m240": neutral,
            "gain": 1.0 / neutral,
            "xy": r["xy"],
            "rounds": r["rounds"],
        }
    mode, cct_in, duv_in = parse_wb_cct_duv(wb_spec)
    if mode == "camera":
        white_scene, cct_scene = src.white_of_neutral(src_neutral)
        label = "camera"
    else:
        white_scene = white_xyz_from_cct_duv(cct_in, duv_in)
        cct_scene = float(cct_in)
        label = f"{int(round(cct_in))}K" if abs(duv_in) < 1e-6 else f"{int(round(cct_in))}K,{duv_in:+.4f}"
    _, duv_scene = cct_duv(white_scene)
    n_m240, _ = adobe.neutral_of_white(white_scene)
    return {
        "mode": label,
        "white_scene": white_scene,
        "cct_scene": float(cct_scene),
        "duv_scene": float(duv_scene),
        "neutral_m240": n_m240,
        "gain": 1.0 / n_m240,
    }


def look_indices(mode: str, contrast: int = 0, saturation: int = 0) -> tuple[int, int]:
    """(mode, contrast, saturation) -> (tone_idx, sat_idx) in the M240 tables."""
    m = mode.strip().lower()
    if m == "standard":
        return int(np.clip(round(contrast), -2, 2)) + 2, int(np.clip(round(saturation), -2, 2)) + 2
    if m == "vivid":
        return 5, 5
    if m == "natural":
        return 6, 6
    raise ValueError(f"Unsupported mode '{mode}'. Choose standard, vivid, or natural.")


def m240_kernel_t(cam12_wb: torch.Tensor, cc_t: torch.Tensor, tone_t: torch.Tensor, gamma_t: torch.Tensor,
                  lumisat_t: torch.Tensor | None) -> torch.Tensor:
    """colorcomp -> tone_gain -> 12-bit gamma -> BT.601 lumisat (None = monochrome); returns sRGB-encoded (N, 3)."""
    rgb = torch.clamp(cam12_wb @ cc_t.T, 0.0, float(TONE_LEN - 1))
    i0 = torch.floor(rgb).long().clamp(0, TONE_LEN - 2)
    frac = rgb - i0.to(torch.float32)
    gain = tone_t[i0] * (1.0 - frac) + tone_t[i0 + 1] * frac
    rgb_t = torch.clamp(rgb * gain, 0.0, float(GAMMA_LEN - 1))

    j0 = torch.floor(rgb_t).long().clamp(0, GAMMA_LEN - 2)
    jfrac = rgb_t - j0.to(torch.float32)
    srgb = gamma_t[j0] * (1.0 - jfrac) + gamma_t[j0 + 1] * jfrac

    y = 0.299 * srgb[..., 0] + 0.587 * srgb[..., 1] + 0.114 * srgb[..., 2]
    if lumisat_t is None:
        return y.unsqueeze(-1).expand_as(srgb)
    yi = torch.clamp(y * float(LUMISAT_LEN - 1), 0.0, float(LUMISAT_LEN - 1))
    k0 = torch.floor(yi).long().clamp(0, LUMISAT_LEN - 2)
    kfrac = yi - k0.to(torch.float32)
    sat_g = lumisat_t[k0] * (1.0 - kfrac) + lumisat_t[k0 + 1] * kfrac
    return torch.clamp(y.unsqueeze(-1) + (srgb - y.unsqueeze(-1)) * sat_g.unsqueeze(-1), 0.0, 1.0)


def render(cam_rgb: np.ndarray, src: DcpProfile, look: dict, wb_state: dict, k_to_m240: float, *,
           mode: str = "standard", contrast: int = 0, saturation: int = 0, mono: bool = False,
           use_hsm: bool = True, out_space: str = "srgb", device: str = "auto",
           chunk_pixels: int = 8_000_000) -> np.ndarray:
    dev = resolve_device(device)
    bridge = DualDcpBridge(src, look["adobe_profile"], wb_state["white_scene"], k_to_m240, use_hsm, dev)
    g_t = as_t(wb_state["gain"], dev)
    cc_t = as_t(look["colorcomp"], dev)
    gamma_t = as_t(look["gamma"], dev)
    t_idx, s_idx = look_indices(mode, contrast, saturation)
    tone_t = as_t(look["tone_gain"][t_idx], dev)
    ls_t = None if mono else as_t(look["lumisat"][s_idx], dev)

    def fn(c):
        cam12_wb = (torch.clamp(bridge(c), min=0.0) * g_t) * M240_WHITE_LEVEL
        return encode_from_srgb_t(m240_kernel_t(cam12_wb, cc_t, tone_t, gamma_t, ls_t), out_space, dev)

    return map_pixels(cam_rgb, fn, dev, chunk_pixels)
