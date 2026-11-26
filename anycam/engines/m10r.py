"""Leica M10-R (firmware 40.26.35.51) engine: external WB -> M10-R gain + CA9 CC0, B2Y kernel (3x3 / Monochrome).

WB closes two loops on the same frame:
  --wb camera        source Adobe CM -> scene white -> M10-R Adobe CM -> gain; the firmware CM_LT/CM_HT
                     (fw profile) turn that gain into CCT_fw, which drives the dynamic CC0.
  --wb CCT[K][,Duv]  firmware CM_LT/CM_HT turn the white into gain and CC0 first; the dual-DCP hand-off
                     white is then adobe_profile.white_of_neutral(1 / gain), so both sides agree.
Exposure: DNG white level 15000, BE_M10R = 0.0.
"""
import struct
from pathlib import Path

import numpy as np
import torch

from ..core.bridge import DualDcpBridge
from ..core.color import BRADFORD, BRADFORD_INV, XYZ_D50, cct_duv, parse_wb_cct_duv, white_xyz_from_cct_duv
from ..core.dcp import DcpProfile
from ..core.gpu import as_t, map_pixels, resolve_device
from ..core.output import encode_from_srgb_t
from ..core.raw import RawFrame

DEFAULT_M10R_LOOK = Path(__file__).resolve().parent.parent.parent / "assets" / "m10r" / "leica_m10r.bin"
BE_M10R = 0.0
M10R_WHITE_LEVEL = 15000.0
TONE_LEN = 10240
DG_LEN = 32768

# Adobe Standard ColorMatrix1/2 of the M10-R (LEICA M10-R Adobe Standard.dcp)
M10R_CM1 = np.array([[0.6341, -0.3301, 0.0255], [-0.4841, 1.2726, 0.2354], [-0.0687, 0.1987, 0.6538]], dtype=np.float64)
M10R_CM2 = np.array([[0.5296, -0.1881, -0.0423], [-0.5483, 1.3367, 0.2327], [-0.1226, 0.2594, 0.555]], dtype=np.float64)


def decode_m10r_bin(path: Path = DEFAULT_M10R_LOOK) -> dict:
    """Decode the Leica M10-R firmware 40.26.35.51 kind-17 look."""
    b = Path(path).read_bytes()
    if b[:8] != b"MCOLLOOK":
        raise ValueError(f"Invalid look magic in {path}")
    ver, kind = struct.unpack("<II", b[8:16])
    if kind != 17:
        raise ValueError(f"Expected kind 17 in {path}, got version={ver}, kind={kind}")
    at = 16
    sat_steps, gray_code, t_lt, t_ht = struct.unpack_from("<Ifff", b, at); at += 16
    mats = []
    for _ in range(6):
        mats.append(np.frombuffer(b, "<f4", 9, at).astype(np.float64).reshape(3, 3).copy())
        at += 36
    fm1, fm2, cm_lt, cm_ht, xyz_to_pro, pro_to_srgb = mats
    tone_weights = np.frombuffer(b, "<f4", 3, at).astype(np.float64).copy(); at += 12
    tone = (np.frombuffer(b, "<u2", 3 * TONE_LEN, at).reshape(3, TONE_LEN).astype(np.float32) / 32768.0).copy(); at += 3 * TONE_LEN * 2
    dg = np.frombuffer(b, "<u2", DG_LEN, at).astype(np.float32).copy(); at += DG_LEN * 2
    yc_matrix = np.frombuffer(b, "<f4", 9, at).astype(np.float64).reshape(3, 3).copy(); at += 36
    sat_scales = np.frombuffer(b, "<f4", sat_steps, at).astype(np.float64).copy(); at += sat_steps * 4
    hd, sd, dcp_t1, dcp_t2 = struct.unpack_from("<IIff", b, at); at += 16
    n = hd * sd * 3
    hs1 = np.frombuffer(b, "<f4", n, at).astype(np.float64).reshape(hd, sd, 3).copy(); at += 4 * n
    hs2 = np.frombuffer(b, "<f4", n, at).astype(np.float64).reshape(hd, sd, 3).copy(); at += 4 * n
    if at != len(b):
        raise ValueError(f"Corrupt look size in {path}")

    adobe_profile = DcpProfile(name="Leica M10-R Adobe Standard", cm=[M10R_CM1, M10R_CM2], fm=[fm1, fm2], hsm=[hs1, hs2], t1=dcp_t1, t2=dcp_t2)
    fw_profile = DcpProfile(name="Leica M10-R firmware CM", cm=[cm_lt, cm_ht], fm=None, t1=t_lt, t2=t_ht)
    return {
        "gray_code": float(gray_code),
        "t_lt": float(t_lt),
        "t_ht": float(t_ht),
        "cm_lt": cm_lt,
        "cm_ht": cm_ht,
        "xyz_to_pro": xyz_to_pro,
        "pro_to_srgb": pro_to_srgb,
        "tone_weights": tone_weights,
        "tone": tone,
        "diff_gamma": dg,
        "yc_matrix": yc_matrix,
        "yc_inv": np.linalg.inv(yc_matrix),
        "sat_scales": sat_scales,
        "adobe_profile": adobe_profile,
        "fw_profile": fw_profile,
    }


def exposure(frame: RawFrame, src: DcpProfile, ev: float) -> float:
    src_base_ev = frame.raw_base_ev(raf_input_scale=True) + src.baseline_ev
    return float(2.0 ** (ev + src_base_ev - BE_M10R))


def compute_cc0(look: dict, white_fw_xyz: np.ndarray, cct_fw: float) -> np.ndarray:
    """CA9 IPC dynamic CC0: white-balanced M10-R camera RGB -> ProPhoto D50 linear."""
    g = float(np.clip((1.0 / cct_fw - 1.0 / look["t_ht"]) / (1.0 / look["t_lt"] - 1.0 / look["t_ht"]), 0.0, 1.0))
    ab = g * look["cm_lt"] + (1.0 - g) * look["cm_ht"]
    cam_w = ab @ white_fw_xyz
    asn = cam_w / cam_w[1]
    ca = BRADFORD_INV @ np.diag((BRADFORD @ white_fw_xyz) / (BRADFORD @ XYZ_D50)) @ BRADFORD
    return look["xyz_to_pro"] @ np.linalg.inv(ab @ ca) @ np.diag(asn)


def solve_wb(src: DcpProfile, src_neutral: np.ndarray, wb_spec: str, look: dict) -> dict:
    adobe = look["adobe_profile"]
    fw = look["fw_profile"]
    mode, cct_in, duv_in = parse_wb_cct_duv(wb_spec)

    if mode == "camera":
        white_scene, cct_scene = src.white_of_neutral(src_neutral)
        _, duv_scene = cct_duv(white_scene)
        n_m10r, _ = adobe.neutral_of_white(white_scene)
        gain = 1.0 / n_m10r
        white_fw, cct_fw = fw.white_of_neutral(n_m10r)
        _, duv_fw = cct_duv(white_fw)
        label = "camera"
    else:
        white_fw = white_xyz_from_cct_duv(cct_in, duv_in)
        asn_g1, cct_fw = fw.neutral_of_white(white_fw)
        _, duv_fw = cct_duv(white_fw)
        gain = 1.0 / asn_g1
        white_scene, cct_scene = adobe.white_of_neutral(asn_g1)
        _, duv_scene = cct_duv(white_scene)
        label = f"{int(round(cct_in))}K" if abs(duv_in) < 1e-6 else f"{int(round(cct_in))}K,{duv_in:+.4f}"

    return {
        "mode": label,
        "white_scene": white_scene,
        "cct_scene": float(cct_scene),
        "duv_scene": float(duv_scene),
        "white_fw": white_fw,
        "cct_fw": float(cct_fw),
        "duv_fw": float(duv_fw),
        "gain": gain,
        "cc0": compute_cc0(look, white_fw, cct_fw),
    }


def m10r_kernel_t(cam14_wb, cc0_t, tone_w_t, tone_curve_t, pro_to_srgb_t, dg_t, yc_mat_t, yc_inv_t, sat_scale: float):
    """B2Y: CC0 -> TONE_CONTROL -> CC1 -> 15-bit DIFF_GAMMA -> YC_CONVERSION; returns sRGB-encoded (N, 3)."""
    pro = torch.clamp(cam14_wb @ cc0_t.T, 0.0, 16383.0)
    y = torch.clamp(pro @ tone_w_t, 0.0, 16383.0)
    t_idx = torch.floor(y * 0.5).long().clamp(0, TONE_LEN - 1)
    pro_t = torch.clamp(pro * tone_curve_t[t_idx].unsqueeze(-1), 0.0, 16383.0)
    lin = torch.clamp(pro_t @ pro_to_srgb_t.T, 0.0, 32767.0)
    i0 = torch.floor(lin).long().clamp(0, DG_LEN - 2)
    frac = lin - i0.to(torch.float32)
    rgb = (dg_t[i0] * (1.0 - frac) + dg_t[i0 + 1] * frac) / 16383.0
    if abs(sat_scale - 1.0) > 1e-6:
        ycc = rgb @ yc_mat_t.T
        ycc[:, 1:] = ycc[:, 1:] * sat_scale
        return torch.clamp(ycc @ yc_inv_t.T, 0.0, 1.0)
    return torch.clamp(rgb, 0.0, 1.0)


def render(cam_rgb: np.ndarray, src: DcpProfile, look: dict, wb_state: dict, k_to_m10r: float, *,
           contrast: int = 0, saturation: int = 0, mono: bool = False, use_hsm: bool = True,
           out_space: str = "srgb", device: str = "auto", chunk_pixels: int = 8_000_000) -> np.ndarray:
    dev = resolve_device(device)
    bridge = DualDcpBridge(src, look["adobe_profile"], wb_state["white_scene"], k_to_m10r, use_hsm, dev)
    g_t = as_t(wb_state["gain"], dev)
    cc0_t = as_t(wb_state["cc0"], dev)
    tone_w_t = as_t(look["tone_weights"], dev)
    pro_to_srgb_t = as_t(look["pro_to_srgb"], dev)
    dg_t = as_t(look["diff_gamma"], dev)
    yc_mat_t = as_t(look["yc_matrix"], dev)
    yc_inv_t = as_t(look["yc_inv"], dev)
    tone_t = as_t(look["tone"][int(np.clip(round(contrast), -1, 1)) + 1], dev)
    sat_idx = 3 if mono else int(np.clip(round(saturation), -1, 1)) + 1
    sat_scale = float(look["sat_scales"][int(np.clip(sat_idx, 0, len(look["sat_scales"]) - 1))])

    def fn(c):
        cam14_wb = (torch.clamp(bridge(c), min=0.0) * g_t) * M10R_WHITE_LEVEL
        rgb = m10r_kernel_t(cam14_wb, cc0_t, tone_w_t, tone_t, pro_to_srgb_t, dg_t, yc_mat_t, yc_inv_t, sat_scale)
        return encode_from_srgb_t(rgb, out_space, dev)

    return map_pixels(cam_rgb, fn, dev, chunk_pixels)
