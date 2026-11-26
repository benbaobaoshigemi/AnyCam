"""Fujifilm X100VI film simulation engine (kind-13 looks, 20 films) on GPU.

Front end: source Adobe Standard DCP -> fixed D65 ForwardMatrix -> ProPhoto D50, frame CCT from the
camera neutral (McCamy on ColorMatrix1/2). HueSatMap alignment: source DCP HueSatMap forward, X100VI
HueSatMap inverse. Then the X100VI pipeline: FRONT_M -> WB shift -> 1:2:1 luma -> tone40 (DR100/200,
Highlight/Shadow) -> film matrix -> tables31 -> chroma4 (Color -4..+4) -> Rec.601 +/-0.5 clamp -> gmod ->
17^3 3D LUT -> huegain + hueres -> Color Chrome Effect & FX Blue -> sRGB / Display P3 / Adobe RGB.

This engine keeps its own HueSatMap kernels (3-D tables, true sRGB curve for sRGB-encoded maps,
V clamped to 1, 16-round full fixed-point inverse); they are part of its calibrated look and differ
from the 2-D bridge kernels in anycam.core.gpu.
"""
import struct
from pathlib import Path

import numpy as np
import torch

from ..core.color import ADOBE_GAMMA, ADOBE_TO_XYZ, BRADFORD, ILLUM_T, SRGB_TO_XYZ, XYZ_D50, XYZ_D65, cat_bradford
from ..core.dcp import read_dcp
from ..core.gpu import resolve_device
from ..core.raw import RawFrame

DEFAULT_LOOKS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fuji"

FUJI_PROPHOTO_TO_XYZ = np.array([
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0000, 0.0000, 0.8249],
], dtype=np.float64)
FUJI_XYZ_TO_PROPHOTO = np.linalg.inv(FUJI_PROPHOTO_TO_XYZ)

def _mccamy(xy):
    n = (xy[0] - 0.3320) / (xy[1] - 0.1858)
    return -449.0 * n ** 3 + 3525.0 * n ** 2 - 6823.3 * n + 5520.33


class FujiDcpProfile:
    """Source Adobe Standard profile as read by the Fuji front end (3-D HueSatMap capable)."""

    def __init__(self, dcp_path: Path):
        d = read_dcp(dcp_path)
        rm = lambda k: (d[k][0::2] / d[k][1::2]).reshape(3, 3)
        self.cm1 = rm(50721)
        self.cm2 = rm(50722) if 50722 in d else self.cm1
        self.t1 = ILLUM_T.get(int(d[50778][0]), 2856.0) if 50778 in d else 2856.0
        self.t2 = ILLUM_T.get(int(d[50779][0]), 6504.0) if 50779 in d else 6504.0
        self.fm1 = rm(50964) if 50964 in d else None
        self.fm2 = rm(50965) if 50965 in d else self.fm1
        self.baseline_ev = float(d.get(51109, [0.0])[0])

        self.hsm = None
        if 50937 in d and 50938 in d:
            hd, sd, vd = (int(x) for x in d[50937])
            shape = (vd, hd, sd, 3)
            hsm1 = d[50938].astype(np.float64).reshape(shape)
            hsm2 = d[50939].astype(np.float64).reshape(shape) if 50939 in d else hsm1
            self.hsm = {
                "dims": (hd, sd, vd),
                "t1": self.t1,
                "t2": self.t2,
                "table1": hsm1,
                "table2": hsm2,
                "srgb_encoded": int(d.get(51107, [0])[0]) == 1,
            }

    def weight(self, cct: float) -> float:
        if abs(self.t2 - self.t1) < 1e-6:
            return 0.5
        return float(np.clip((1.0 / cct - 1.0 / self.t1) / (1.0 / self.t2 - 1.0 / self.t1), 0.0, 1.0))

    def cm(self, cct: float) -> np.ndarray:
        w = self.weight(cct)
        return self.cm1 * (1.0 - w) + self.cm2 * w

    def forward_matrix(self, cct: float) -> np.ndarray | None:
        if self.fm1 is None:
            return None
        w = self.weight(cct)
        return self.fm1 * (1.0 - w) + self.fm2 * w

    def white_from_neutral(self, neutral: np.ndarray) -> tuple[tuple[float, float], float]:
        xy = (0.3457, 0.3585)
        lo, hi = min(self.t1, self.t2), max(self.t1, self.t2)
        for _ in range(100):
            t = float(np.clip(_mccamy(xy), lo, hi))
            xyz = np.linalg.inv(self.cm(t)) @ neutral
            s = xyz.sum()
            new = (float(xyz[0] / s), float(xyz[1] / s))
            done = abs(new[0] - xy[0]) + abs(new[1] - xy[1]) < 1e-8
            xy = new
            if done:
                break
        return xy, float(np.clip(_mccamy(xy), lo, hi))

    def hsm_table(self, cct: float):
        if self.hsm is None:
            return None
        w = self.weight(cct)
        return self.hsm["table1"] * (1.0 - w) + self.hsm["table2"] * w


def fuji_scene(frame: RawFrame, dcp_path: Path, ev: float = 0.0, cct_override: float | None = None):
    """Camera RGB -> ProPhoto D50 matrix (fixed D65 ForwardMatrix x camera WB x exposure) and frame CCT."""
    prof = FujiDcpProfile(dcp_path)
    neutral = frame.neutral
    raw_base_ev = frame.raw_base_ev(raf_input_scale=False)
    xy, cct = prof.white_from_neutral(neutral)
    if cct_override is not None:
        cct = float(cct_override)

    wb_gain = (1.0 / neutral).astype(np.float64)
    # D65 ForwardMatrix (fm2) is the fixed sensor->ProPhoto calibration, matching the fixed matrix
    # inside the X100VI ISP & FRONT_M without tungsten FM1 over-saturation.
    fm = prof.fm2 if prof.fm2 is not None else prof.fm1
    if fm is not None:
        cam_to_pp = FUJI_XYZ_TO_PROPHOTO @ fm
    else:
        w_xyz = np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]], dtype=np.float64)
        inv_cm = np.linalg.inv(prof.cm(cct))
        s = 1.0 / (inv_cm @ neutral)[1]
        cam_to_xyz_d50 = cat_bradford(w_xyz, XYZ_D50) @ (s * inv_cm)
        cam_to_pp = FUJI_XYZ_TO_PROPHOTO @ cam_to_xyz_d50

    total_gain = float(2.0 ** (ev + prof.baseline_ev + raw_base_ev))
    m_total = (cam_to_pp @ np.diag(wb_gain) * total_gain).astype(np.float32)
    return m_total, {
        "cct": cct,
        "base_ev": raw_base_ev + prof.baseline_ev,
        "profile": prof,
        "hsm_table": prof.hsm_table(cct),
        "hsm_srgb": prof.hsm["srgb_encoded"] if prof.hsm is not None else False,
    }


X100VI_FILMS = [
    "provia", "velvia", "astia", "classic_chrome", "reala_ace",
    "pro_neg_hi", "pro_neg_std", "classic_neg", "nostalgic_neg",
    "eterna", "bleach_bypass",
    "acros", "acros_ye", "acros_r", "acros_g",
    "monochrome", "monochrome_ye", "monochrome_r", "monochrome_g",
    "sepia",
]


def _f32(buf, off, count):
    return np.frombuffer(buf, "<f4", count, off).astype(np.float64), off + 4 * count


def decode_look_bin(path: Path) -> dict:
    """Decode an x100vi_*.bin look file (MCOLLOOK v1, kind 13) matching x100vi_engine.decode."""
    buf = Path(path).read_bytes()
    if buf[:8] != b"MCOLLOOK":
        raise ValueError(f"Invalid look magic in {path}")
    ver, kind = struct.unpack("<II", buf[8:16])
    if kind != 13:
        raise ValueError(f"Expected kind 13 in {path}, got version={ver}, kind={kind}")
    at = 16
    scale = struct.unpack_from("<f", buf, at)[0]
    at += 4
    tone, at = _f32(buf, at, 40)
    curves_flat, at = _f32(buf, at, 3 * 31)
    edges, at = _f32(buf, at, 11)
    matrix_flat, at = _f32(buf, at, 9)
    color_flat, at = _f32(buf, at, 9)
    luma, at = _f32(buf, at, 3)
    cube_flat, at = _f32(buf, at, 17 * 17 * 17 * 3)

    shadows_flat, at = _f32(buf, at, 2 * 7 * 40)
    gm, at = _f32(buf, at, 14)
    hg_len = 1 + 36 * 5 + 3 * 36 * 6 * 8
    hg_all, at = _f32(buf, at, hg_len)

    hd, sd = struct.unpack_from("<2I", buf, at)
    at += 8
    t1, t2 = struct.unpack_from("<2f", buf, at)
    at += 8
    n = hd * sd * 3
    tb1, at = _f32(buf, at, n)
    tb2, at = _f32(buf, at, n)
    if at != len(buf):
        raise ValueError(f"Trailing bytes in {path}: read {at} != {len(buf)}")

    gmod_is_identity = (not gm[:4].any()) and (not gm[4:9].any()) and np.allclose(gm[9:], 256.0)
    has_hg = hg_all[0] != 0.0
    p = {
        "scale": float(scale),
        "tone": tone,
        "curves": curves_flat.reshape(3, 31),
        "edges": edges,
        "matrix": matrix_flat.reshape(3, 3),
        "color": color_flat.reshape(3, 3),
        "luma": luma,
        "cube": cube_flat.reshape(17, 17, 17, 3),
        "shadows": shadows_flat.reshape(2, 7, 40),
        "gmod": None if gmod_is_identity else gm,
        "huegain": hg_all[1:1 + 36 * 5].reshape(36, 5) if has_hg else None,
        "hueres": hg_all[1 + 36 * 5:].reshape(3, 36, 6, 8) if has_hg else None,
        "hsm": {
            "dims": (hd, sd, 1),
            "t1": float(t1),
            "t2": float(t2),
            "table1": tb1.reshape(1, hd, sd, 3),
            "table2": tb2.reshape(1, hd, sd, 3),
        },
        "mode_d": Path(path).stem == "x100vi_reala_ace",
    }
    return p


def load_film_look(film: str, looks_dir: Path = DEFAULT_LOOKS_DIR) -> dict:
    name = film.strip().lower().replace("-", "_").replace(" ", "_")
    if name not in X100VI_FILMS:
        raise ValueError(f"Unknown film '{film}'. Available: {', '.join(X100VI_FILMS)}")
    path = Path(looks_dir) / f"x100vi_{name}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Look file not found: {path}")
    return decode_look_bin(path)


FRONT_GAIN = 0.996683
FRONT_PEDESTAL = 0.000123

FRONT_M = np.array([
    [ 1.175569, -0.086910, -0.088659],
    [-0.026913,  0.990242,  0.036671],
    [-0.132675,  0.143012,  0.989663],
], dtype=np.float64)

_CAT_D50_D65 = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ XYZ_D65) / (BRADFORD @ XYZ_D50)) @ BRADFORD
PP_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ) @ _CAT_D50_D65 @ FUJI_PROPHOTO_TO_XYZ

ADOBE_TO_P3 = np.array([
    [1.15015701, -0.15009213, -0.00003531],
    [0.04641974,  0.95356952,  0.00000245],
    [0.02388714,  0.02650547,  0.94960698],
], dtype=np.float64)

ADOBE_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ) @ ADOBE_TO_XYZ

_FW_YCC = np.array([[306, 601, 117], [-173, -339, 512], [512, -429, -83]], dtype=np.float64) / 1024.0
_FW_YCC_INV = np.linalg.inv(_FW_YCC)

WB_SHIFT_R = np.array([554, 641, 735, 782, 834, 892, 913, 945, 977, 1024, 1050, 1088, 1124, 1159, 1242, 1315, 1417, 1612, 1896], dtype=np.float64) / 1024.0
WB_SHIFT_B = np.array([587, 665, 755, 800, 852, 904, 932, 953, 990, 1024, 1041, 1064, 1095, 1120, 1169, 1225, 1278, 1398, 1534], dtype=np.float64) / 1024.0
COLOR_GAIN = np.array([102, 109, 115, 122, 128, 134, 141, 147, 154], dtype=np.float64) / 128.0
TONE_EDGES = np.array([0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384], dtype=np.float64)

HUERES_YK = np.array([0.10, 0.25, 0.35, 0.45, 0.55, 0.70], dtype=np.float64)
HUERES_CK = np.array([0.0, 0.08, 0.14, 0.20, 0.24, 0.27, 0.30, 0.34], dtype=np.float64)
HUEGAIN_HUES, HUEGAIN_YS = 36, 5
HUEGAIN_Y0, HUEGAIN_DY = 0.075, 0.18125

_CCE_YCC = np.array([
    [ 0.2988,  0.5869,  0.1143],
    [-0.1689, -0.3311,  0.5000],
    [ 0.5000, -0.4189, -0.0811],
], dtype=np.float64)
CCE_A_Y = 0.8452
CCE_C_CH = 0.3693
CCE_X_E = (159.0924 - 162.7657, 159.0924)
CCE_X_B = (215.0538 - 50.2728, 215.0538)
CCE_A = (-0.3167, -0.4593, 155.2244, 147.0820, 550.9013, 518.9044)
CCE_HUE_OFFSET = 1.5509
_CCE_EDGES_E = np.array([47.0, 83.0, 156.0, 174.0, 286.0, 347.0], dtype=np.float64)
_CCE_VECS_E = np.array([[-0.4909, 0.1811], [-0.8408, 0.5362], [-1.0484, 0.1238], [0.0000, 0.0000], [0.2013, 0.4390], [-0.3056, 0.7095]], dtype=np.float64)
_CCE_EDGES_B = np.array([47.0, 83.0, 168.0, 189.0, 260.0, 295.0, 355.0], dtype=np.float64)
_CCE_VECS_B = np.array([[0.0000, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000], [1.3373, -0.2973], [0.6737, 0.0068], [0.2588, -0.0512], [0.0000, 0.0000]], dtype=np.float64)
MODE_D_A = (96 / 256, 176 / 256, 1.0)
MODE_D_R = (32 / 256, 144 / 256, 1.0)
MODE_D_B = (96 / 256, 176 / 256, 1.0)
_MODE_D_R_ZONES = (0, 1, 2)


def _t(arr, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _rgb_to_hsv_t(rgb: torch.Tensor):
    mx, _ = rgb.max(dim=-1)
    mn, _ = rgb.min(dim=-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ds = torch.where(d > 1e-12, d, torch.ones_like(d))
    h = torch.where(
        mx == r,
        torch.remainder((g - b) / ds, 6.0),
        torch.where(mx == g, (b - r) / ds + 2.0, (r - g) / ds + 4.0),
    )
    h = torch.where(d > 1e-12, h, torch.zeros_like(h))
    s = torch.where(mx > 1e-12, d / torch.where(mx > 1e-12, mx, torch.ones_like(mx)), torch.zeros_like(mx))
    return torch.remainder(h, 6.0), s, mx


def _hsv_to_rgb_t(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    i = torch.remainder(torch.floor(h).to(torch.int64), 6)
    f = h - torch.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack((r, g, b), dim=-1)


def fuji_apply_hsm_t(rgb: torch.Tensor, table: torch.Tensor, srgb_encoded: bool = False) -> torch.Tensor:
    """Apply Adobe ProfileHueSatMap (vd, hd, sd, 3) on (N, 3) tensor."""
    vd, hd, sd, _ = table.shape
    enc = rgb
    if srgb_encoded:
        enc = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * torch.clamp(rgb, min=1e-9) ** (1.0 / 2.4) - 0.055)
    h, s, v = _rgb_to_hsv_t(torch.clamp(enc, min=0.0))
    hs = h * (hd / 6.0)
    ss = s * (sd - 1)
    h0 = torch.remainder(torch.floor(hs).to(torch.int64), hd)
    hf = hs - torch.floor(hs)
    h1 = torch.remainder(h0 + 1, hd)
    s0 = torch.clamp(torch.floor(ss).to(torch.int64), max=sd - 2)
    sf = ss - s0
    if vd > 1:
        vs = v * (vd - 1)
        v0 = torch.clamp(torch.floor(vs).to(torch.int64), max=vd - 2)
        vf = vs - v0
    else:
        v0 = torch.zeros_like(h0)
        vf = torch.zeros_like(v)
    v1 = torch.clamp(v0 + 1, max=vd - 1)

    out = torch.zeros_like(enc)
    for vi, vw in ((v0, 1.0 - vf), (v1, vf)):
        for hi, hw in ((h0, 1.0 - hf), (h1, hf)):
            for si, sw in ((s0, 1.0 - sf), (s0 + 1, sf)):
                out = out + (vw * hw * sw).unsqueeze(-1) * table[vi, hi, si]
    h2 = torch.remainder(h + out[..., 0] / 60.0, 6.0)
    s2 = torch.clamp(s * out[..., 1], 0.0, 1.0)
    v2 = torch.clamp(v * out[..., 2], 0.0, 1.0)
    res = _hsv_to_rgb_t(h2, s2, v2)
    if srgb_encoded:
        res = torch.where(res <= 0.04045, res / 12.92, ((res + 0.055) / 1.055) ** 2.4)
    return res


def fuji_inverse_hsm_t(s: torch.Tensor, table: torch.Tensor, rounds: int = 16) -> torch.Tensor:
    s = torch.clamp(s, min=0.0)
    h1, s1, v1 = _rgb_to_hsv_t(s)
    a = s.clone()
    for _ in range(rounds):
        ha, sa, va = _rgb_to_hsv_t(a)
        hf, sf, vf = _rgb_to_hsv_t(fuji_apply_hsm_t(a, table, srgb_encoded=False))
        dh = torch.remainder(hf - ha + 3.0, 6.0) - 3.0
        ms = torch.where(sa > 1e-4, sf / torch.clamp(sa, min=1e-4), torch.ones_like(sa))
        mv = torch.where(va > 1e-6, vf / torch.clamp(va, min=1e-6), torch.ones_like(va))
        ms = torch.where(ms > 1e-3, ms, torch.ones_like(ms))
        mv = torch.where(mv > 1e-3, mv, torch.ones_like(mv))
        a = _hsv_to_rgb_t(h1 - dh, torch.clamp(s1 / ms, 0.0, 1.0), v1 / mv)
    return a


def _lookup_inv_t(h: torch.Tensor, s: torch.Tensor, inv: torch.Tensor):
    hd, sd, _ = inv.shape
    hs, ss = h * (hd / 6.0), s * (sd - 1)
    h0 = torch.floor(hs)
    hf = (hs - h0).unsqueeze(-1)
    h0 = torch.remainder(h0.long(), hd)
    h1 = torch.remainder(h0 + 1, hd)
    s0 = torch.floor(ss).long().clamp(0, sd - 2)
    sf = (ss - s0).unsqueeze(-1)
    m = (inv[h0, s0] * (1.0 - hf) + inv[h1, s0] * hf) * (1.0 - sf) + (inv[h0, s0 + 1] * (1.0 - hf) + inv[h1, s0 + 1] * hf) * sf
    return m[..., 0], m[..., 1], m[..., 2]


def fuji_inverse_hsm_fast_t(s: torch.Tensor, table: torch.Tensor, inv, safe_v: float) -> torch.Tensor:
    """fuji_inverse_hsm_t through the build_fuji_inv_table lookup.

    Pixels that can hit the V clamp in some round (v >= safe_v), are too dark for the V ratio guard,
    or fall in a rough cell run the full 16-round original.
    """
    s = torch.clamp(s, min=0.0)
    h1, s1, v1 = _rgb_to_hsv_t(s)
    inv, rough = inv
    dh0, sa0, vr0 = _lookup_inv_t(h1, s1, inv)
    a = _hsv_to_rgb_t(torch.remainder(h1 + dh0, 6.0), sa0, v1 * vr0)
    hd, sd, _ = inv.shape
    cell = rough[torch.remainder(torch.floor(h1 * (hd / 6.0)).long(), hd), torch.floor(s1 * (sd - 1)).long().clamp(0, sd - 2)]
    slow = (v1 >= safe_v) | (v1 <= 1e-5) | cell
    if bool(slow.any()):
        a[slow] = fuji_inverse_hsm_t(s[slow], table)
    return a


def _inv_samples(table: torch.Tensor, h: torch.Tensor, s: torch.Tensor, rounds: int) -> torch.Tensor:
    rgb = _hsv_to_rgb_t(h, s, torch.full_like(h, 0.5))
    ha, sa, va = _rgb_to_hsv_t(fuji_inverse_hsm_t(rgb, table, rounds=rounds))
    h_in, _, v_in = _rgb_to_hsv_t(rgb)
    dh = torch.remainder(ha - h_in + 3.0, 6.0) - 3.0
    return torch.stack((dh, sa, va / v_in), -1)


def build_fuji_inv_table(table: torch.Tensor, hd_inv: int = 1440, sd_inv: int = 513, rounds: int = 16,
                         tol: float = 2e-5):
    """Tabulate the 16-round inverse per input (h, s): (hd_inv, sd_inv, 3) = (hue shift, saturation, V ratio).

    For a 2-D map the H/S recursion of fuji_inverse_hsm_t does not depend on V while no V clamp
    occurs, so the table is built at V = 0.5. The inverse is not smooth everywhere (the 16 rounds
    have not settled on some colours): every cell is checked at its centre against the exact inverse,
    and cells whose bilinear value misses by more than `tol` are flagged (hd_inv, sd_inv) for the
    exact path.
    """
    dev = table.device
    hg = torch.arange(hd_inv, device=dev, dtype=torch.float32) * (6.0 / hd_inv)
    sg = torch.linspace(0.0, 1.0, sd_inv, device=dev, dtype=torch.float32)
    hh, ss = torch.meshgrid(hg, sg, indexing="ij")
    inv = _inv_samples(table, hh.reshape(-1), ss.reshape(-1), rounds).reshape(hd_inv, sd_inv, 3)

    hc = hh[:, :-1] + 3.0 / hd_inv
    sc = ss[:, :-1] + 0.5 / (sd_inv - 1)
    exact = _inv_samples(table, hc.reshape(-1), sc.reshape(-1), rounds)
    approx = torch.stack(_lookup_inv_t(hc.reshape(-1), sc.reshape(-1), inv), -1)
    err = (exact - approx).abs().amax(-1).reshape(hd_inv, sd_inv - 1)
    rough = torch.zeros((hd_inv, sd_inv), dtype=torch.bool, device=dev)
    rough[:, :-1] = err > tol
    return inv, rough


def _bezier_t(values: torch.Tensor, edges: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    k = torch.clamp(torch.searchsorted(edges, x, side="right") - 1, 0, edges.shape[0] - 2)
    e0 = edges[k]
    e1 = edges[k + 1]
    t = (x - e0) / (e1 - e0)
    s = 1.0 - t
    v0 = values[k * 3]
    v1 = values[k * 3 + 1]
    v2 = values[k * 3 + 2]
    v3 = values[k * 3 + 3]
    return v0 * (s ** 3) + 3.0 * v1 * t * (s ** 2) + 3.0 * v2 * (t ** 2) * s + v3 * (t ** 3)


def _interp1d_t(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear 1D interpolation matching np.interp on sorted xp."""
    idx = torch.clamp(torch.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    t = torch.clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return fp[idx] * (1.0 - t) + fp[idx + 1] * t


def _knot_pos_t(knots: torch.Tensor, v: torch.Tensor):
    idx_grid = torch.arange(knots.shape[0], device=knots.device, dtype=knots.dtype)
    p = torch.clamp(_interp1d_t(v, knots, idx_grid), 0.0, float(knots.shape[0] - 1))
    k0 = torch.clamp(torch.floor(p).to(torch.int64), max=knots.shape[0] - 2)
    return k0, p - k0


def _resolve_tone_np(p, highlight=0.0, shadow=0.0, dr=100):
    rows = p["shadows"][1 if dr == 200 else 0]
    def interp_c40(val):
        v = float(np.clip(val, -2.0, 4.0)) + 2.0
        idx0 = min(int(np.floor(v)), 5)
        frac = v - idx0
        return rows[idx0] * (1.0 - frac) + rows[idx0 + 1] * frac
    sh = interp_c40(shadow)
    if highlight == shadow:
        return sh
    hi = interp_c40(highlight)
    return np.r_[sh[:24], hi[24:]]


def _apply_cce_t(rgb: torch.Tensor, ef: int, bf: int, mode_d: bool, dev: torch.device) -> torch.Tensor:
    if ef == 0 and bf == 0 and not mode_d:
        return rgb
    cce_ycc = _t(_CCE_YCC, dev)
    ycc = (rgb @ cce_ycc.T) * 255.0
    mn, _ = rgb.min(dim=-1)
    mn = mn * 255.0
    c = torch.hypot(ycc[..., 1], ycc[..., 2])
    x = CCE_A_Y * ycc[..., 0] + (1.0 - CCE_A_Y) * mn - CCE_C_CH * c
    h = torch.remainder(torch.rad2deg(torch.atan2(-ycc[..., 1], ycc[..., 2])) + 720.0 - CCE_HUE_OFFSET, 360.0)

    edges_e = _t(_CCE_EDGES_E, dev)
    vecs_e = _t(_CCE_VECS_E, dev)
    edges_b = _t(_CCE_EDGES_B, dev)
    vecs_b = _t(_CCE_VECS_B, dev)

    def eval_sector(edges, vecs, x_min, x_max, scale=None):
        idx0 = torch.searchsorted(edges, h, side="right") - 1
        idx0 = torch.where(idx0 < 0, torch.full_like(idx0, edges.shape[0] - 1), idx0)
        idx1 = (idx0 + 1) % edges.shape[0]
        span = torch.remainder(edges[idx1] - edges[idx0], 360.0)
        t = torch.remainder(h - edges[idx0], 360.0) / span
        f = torch.clamp((x_max - x) / (x_max - x_min), 0.0, 1.0)
        cb, cr = ycc[..., 1], ycc[..., 2]
        s_t = torch.ones_like(edges) if scale is None else scale
        return ((1.0 - t) * s_t[idx0] * (vecs[idx0, 0] * cb + vecs[idx0, 1] * cr)
                + t * s_t[idx1] * (vecs[idx1, 0] * cb + vecs[idx1, 1] * cr)) * f

    if mode_d:
        zone = torch.full((edges_e.shape[0],), MODE_D_A[ef], device=dev, dtype=torch.float32)
        for k in _MODE_D_R_ZONES:
            zone[k] = MODE_D_R[ef]
        qe = eval_sector(edges_e, vecs_e, CCE_X_E[0], CCE_X_E[1], zone)
        qb = eval_sector(edges_b, vecs_b, CCE_X_B[0], CCE_X_B[1])
        fe = torch.clamp((CCE_A[2] - x) / CCE_A[3], 0.0, 1.0)
        qe_a = (CCE_A[0] * ycc[..., 1] + CCE_A[1] * ycc[..., 2]) * fe * torch.clamp((CCE_A[4] - h) / CCE_A[5], 0.0, 1.0)
        in_a = (h >= 134.0) & (h <= 194.2)
        we = torch.where(h < 134.0, 1.0, torch.where(h < 273.9, 0.0, torch.clamp((h - 273.9) / (307.6 - 273.9), 0.0, 1.0)))
        wb = torch.where(h < 134.0, 0.0, torch.where(h <= 278.5, 1.0, torch.clamp((318.9 - h) / (318.9 - 278.5), 0.0, 1.0)))
        q = torch.where(in_a, MODE_D_A[ef] * qe_a, we * qe) + MODE_D_B[bf] * wb * qb
    else:
        qe = eval_sector(edges_e, vecs_e, CCE_X_E[0], CCE_X_E[1]) if ef > 0 else 0.0
        qb = eval_sector(edges_b, vecs_b, CCE_X_B[0], CCE_X_B[1]) if bf > 0 else 0.0
        if bf == 0:
            q = (ef * 0.5) * qe
        elif ef == 0:
            q = (bf * 0.5) * qb
        else:
            fe = torch.clamp((CCE_A[2] - x) / CCE_A[3], 0.0, 1.0)
            qe_a = (CCE_A[0] * ycc[..., 1] + CCE_A[1] * ycc[..., 2]) * fe * torch.clamp((CCE_A[4] - h) / CCE_A[5], 0.0, 1.0)
            in_a = (h >= 134.0) & (h <= 194.2)
            we = torch.where(h < 134.0, 1.0, torch.where(h < 273.9, 0.0, torch.clamp((h - 273.9) / (307.6 - 273.9), 0.0, 1.0)))
            wb = torch.where(h < 134.0, 0.0, torch.where(h <= 278.5, 1.0, torch.clamp((318.9 - h) / (318.9 - 278.5), 0.0, 1.0)))
            q = (ef * 0.5) * torch.where(in_a, qe_a, we * qe) + (bf * 0.5) * wb * qb
    return torch.clamp(rgb - torch.clamp(q, min=0.0).unsqueeze(-1) / 255.0, 0.0, 1.0)


def _encode_output_t(out_adobe: torch.Tensor, out_space: str, dev: torch.device) -> torch.Tensor:
    """Convert Adobe RGB encoded [0, 1] core output to target display space encoded [0, 1]."""
    out_adobe = torch.clamp(out_adobe, 0.0, 1.0)
    if out_space == "adobe":
        return out_adobe
    lin = out_adobe ** ADOBE_GAMMA
    if out_space == "p3":
        m = _t(ADOBE_TO_P3, dev)
        lin_dst = torch.clamp(lin @ m.T, 0.0, 1.0)
        return torch.where(lin_dst <= 0.0031308, lin_dst * 12.92, 1.055 * torch.clamp(lin_dst, min=1e-9) ** (1.0 / 2.4) - 0.055)
    if out_space == "srgb":
        m = _t(ADOBE_TO_SRGB, dev)
        lin_dst = torch.clamp(lin @ m.T, 0.0, 1.0)
        return torch.where(lin_dst <= 0.0031308, lin_dst * 12.92, 1.055 * torch.clamp(lin_dst, min=1e-9) ** (1.0 / 2.4) - 0.055)
    raise ValueError(f"Unknown out_space: {out_space}")


def render_image_gpu(
    cam_rgb: np.ndarray,
    cam_to_pp: np.ndarray,
    look: dict,
    cct: float = 5500.0,
    src_hsm_table: np.ndarray | None = None,
    src_hsm_srgb: bool = False,
    use_hsm: bool = True,
    dr: int = 100,
    highlight: float = 0.0,
    shadow: float = 0.0,
    color_step: int = 0,
    wb_r: int = 0,
    wb_b: int = 0,
    color_chrome: int = 0,
    color_chrome_blue: int = 0,
    out_space: str = "srgb",
    device: str = "auto",
    chunk_pixels: int = 2_000_000,
    exact_inverse: bool = False,
) -> np.ndarray:
    """Render full-resolution camera RGB through DCP ForwardMatrix + X100VI engine on GPU.

    exact_inverse=False inverts the X100VI HueSatMap through a per-frame table of the 16-round
    inverse (bright / near-black pixels still run the 16 rounds); True runs 16 rounds everywhere.
    """
    dev = resolve_device(device)
    H, W, _ = cam_rgb.shape
    flat_cam = cam_rgb.reshape(-1, 3)
    N = flat_cam.shape[0]
    out_host = np.empty((N, 3), dtype=np.float32)

    # Pre-upload constant tensors once
    m_cam_pp = _t(cam_to_pp, dev)
    m_pp_srgb = _t(PP_TO_SRGB, dev)
    m_front = _t(FRONT_M, dev)
    m_film = _t(look["matrix"], dev)
    m_fw_ycc = _t(_FW_YCC, dev)
    m_fw_ycc_inv = _t(_FW_YCC_INV, dev)
    tone_t = _t(_resolve_tone_np(look, highlight, shadow, dr), dev)
    tone_edges_t = _t(TONE_EDGES, dev)
    curves_t = _t(look["curves"], dev)
    edges_t = _t(look["edges"], dev)
    cube_t = _t(look["cube"], dev)

    cg = COLOR_GAIN[int(np.clip(round(color_step), -4, 4)) + 4]
    color_mat = _t(look["luma"][None, :] + cg * (look["color"] - look["luma"][None, :]), dev)
    wb_mul = _t([WB_SHIFT_R[int(np.clip(round(wb_r), -9, 9)) + 9], 1.0, WB_SHIFT_B[int(np.clip(round(wb_b), -9, 9)) + 9]], dev)

    src_hsm_t = _t(src_hsm_table, dev) if (use_hsm and src_hsm_table is not None) else None
    dst_hsm_t = None
    if use_hsm and "hsm" in look:
        hsm = look["hsm"]
        w = float(np.clip((1.0 / (cct or 5000.0) - 1.0 / hsm["t1"]) / (1.0 / hsm["t2"] - 1.0 / hsm["t1"]), 0.0, 1.0))
        dst_hsm_t = _t(hsm["table1"] * (1.0 - w) + hsm["table2"] * w, dev)
    inv_t = None
    if dst_hsm_t is not None and not exact_inverse and dst_hsm_t.shape[0] == 1:
        inv_t = build_fuji_inv_table(dst_hsm_t)
        safe_v = float(dst_hsm_t[..., 2].min() / dst_hsm_t[..., 2].max()) * 0.999

    gm_t = _t(look["gmod"], dev) if look.get("gmod") is not None else None
    hg_t = _t(look["huegain"], dev) if look.get("huegain") is not None else None
    hr_t = _t(look["hueres"], dev) if look.get("hueres") is not None else None
    hr_yk = _t(HUERES_YK, dev)
    hr_ck = _t(HUERES_CK, dev)

    ef = int(np.clip(round(color_chrome), 0, 2))
    bf = int(np.clip(round(color_chrome_blue), 0, 2))
    mode_d = bool(look.get("mode_d", False))
    scale = float(look["scale"])

    with torch.no_grad():
        for start in range(0, N, chunk_pixels):
            end = min(start + chunk_pixels, N)
            c = torch.from_numpy(flat_cam[start:end]).to(device=dev, dtype=torch.float32)
            # 1. Target camera RGB -> ProPhoto D50 (ForwardMatrix)
            pp = c @ m_cam_pp.T
            # 2. Positive transform: Target DCP HueSatMap (if enabled)
            if src_hsm_t is not None:
                pp = fuji_apply_hsm_t(pp, src_hsm_t, srgb_encoded=src_hsm_srgb)
            # 3. Inverse transform: Invert X100VI HueSatMap -> X100VI ForwardMatrix-only scene
            if dst_hsm_t is not None:
                if inv_t is not None:
                    pp = fuji_inverse_hsm_fast_t(pp, dst_hsm_t, inv_t, safe_v)
                else:
                    pp = fuji_inverse_hsm_t(pp, dst_hsm_t)

            # 4. Front stage -> working space
            lin = (pp @ m_pp_srgb.T - FRONT_PEDESTAL) * FRONT_GAIN
            lin = lin * wb_mul
            rgb = torch.clamp(lin @ m_front.T, min=0.0) * scale

            # 5. 1:2:1 luma & tone40
            lum = torch.clamp(0.25 * rgb[:, 0] + 0.5 * rgb[:, 1] + 0.25 * rgb[:, 2], min=0.0)
            x = torch.clamp(lum * 16384.0, 0.0, 16384.0)
            output = (x / 16384.0) * _bezier_t(tone_t, tone_edges_t, x)
            gain = torch.where(lum > 0.0, output / torch.clamp(lum, min=1e-12), torch.ones_like(lum))
            if dr == 200:
                t_last = tone_t[-1]
                t_prev = tone_t[-2]
                ext_gain = t_last * (torch.clamp(lum, min=1e-9) ** (12.0 * (t_last - t_prev) / t_last))
                gain = torch.where(lum > 1.0, ext_gain, gain)

            # 6. Film matrix & tables31
            mixed = torch.clamp((rgb * gain.unsqueeze(-1)) @ m_film.T, min=0.0)
            enc_ch = [_bezier_t(curves_t[i], edges_t, torch.clamp(mixed[:, i] * 16384.0, 0.0, 16384.0)) for i in range(3)]
            encoded = torch.stack(enc_ch, dim=1)

            # 7. chroma4 & Rec.601 +/-0.5 clamp
            ycc = (encoded @ color_mat.T) @ m_fw_ycc.T
            ycc[:, 1] = torch.clamp(ycc[:, 1], -0.5, 0.5)
            ycc[:, 2] = torch.clamp(ycc[:, 2], -0.5, 0.5)
            code = ycc @ m_fw_ycc_inv.T

            # 8. gmod
            if gm_t is not None:
                luma = code @ m_fw_ycc[0]
                yy = torch.clamp(luma, 0.0, 1.0) * 255.0
                gi = torch.searchsorted(gm_t[:4], yy, side="right")
                gg = (gm_t[9:][gi] + gm_t[4:9][gi] * yy / 256.0) / 256.0
                code = luma.unsqueeze(-1) + gg.unsqueeze(-1) * (code - luma.unsqueeze(-1))

            # 9. 17^3 3D LUT trilinear interpolation (LUT_CODE_MAX = 1.5)
            code = torch.clamp(code, 0.0, 1.5) * 4095.0
            base = torch.clamp(torch.floor(code / 256.0).to(torch.int64), max=15)
            denom = torch.where(base == 15, torch.full_like(code, 255.0), torch.full_like(code, 256.0))
            frac = (code - base.to(torch.float32) * 256.0) / denom
            out = torch.zeros_like(code)
            fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
            bx, by, bz = base[:, 0], base[:, 1], base[:, 2]
            for zz in (0, 1):
                wz = fz if zz else (1.0 - fz)
                for yy in (0, 1):
                    wy = fy if yy else (1.0 - fy)
                    for xx in (0, 1):
                        wx = fx if xx else (1.0 - fx)
                        out = out + (wx * wy * wz).unsqueeze(-1) * cube_t[bz + zz, by + yy, bx + xx]

            # 10. Hue gain & high-chroma residual
            if hg_t is not None:
                luma = out @ m_fw_ycc[0]
                cb = out @ m_fw_ycc[1]
                cr = out @ m_fw_ycc[2]
                hue = torch.remainder(torch.rad2deg(torch.atan2(-cr, -cb)), 360.0)
                hp = hue / 10.0 - 0.5
                i0f = torch.floor(hp)
                i0 = torch.remainder(i0f.to(torch.int64), HUEGAIN_HUES)
                i1 = torch.remainder(i0 + 1, HUEGAIN_HUES)
                fh = hp - i0f
                yp = torch.clamp((luma - HUEGAIN_Y0) / HUEGAIN_DY, 0.0, float(HUEGAIN_YS - 1))
                j0 = torch.clamp(torch.floor(yp).to(torch.int64), max=HUEGAIN_YS - 2)
                fy_h = yp - j0
                g_hg = (
                    hg_t[i0, j0] * (1.0 - fh) * (1.0 - fy_h)
                    + hg_t[i1, j0] * fh * (1.0 - fy_h)
                    + hg_t[i0, j0 + 1] * (1.0 - fh) * fy_h
                    + hg_t[i1, j0 + 1] * fh * fy_h
                )
                chroma0 = torch.hypot(cb, cr)
                w = torch.clamp((chroma0 - 0.03) / 0.05, 0.0, 1.0)
                g_hg = 1.0 + w * (g_hg - 1.0)
                out = luma.unsqueeze(-1) + g_hg.unsqueeze(-1) * (out - luma.unsqueeze(-1))

                if hr_t is not None:
                    luma = out @ m_fw_ycc[0]
                    cb = out @ m_fw_ycc[1]
                    cr = out @ m_fw_ycc[2]
                    chroma = torch.hypot(cb, cr)
                    hue = torch.remainder(torch.rad2deg(torch.atan2(-cr, -cb)), 360.0)
                    hp = hue / 10.0 - 0.5
                    i0f = torch.floor(hp)
                    i0 = torch.remainder(i0f.to(torch.int64), HUEGAIN_HUES)
                    i1 = torch.remainder(i0 + 1, HUEGAIN_HUES)
                    fh = hp - i0f
                    j0, fy_r = _knot_pos_t(hr_yk, luma)
                    k0, fc_r = _knot_pos_t(hr_ck, chroma)
                    delta = torch.zeros((3, out.shape[0]), device=dev, dtype=torch.float32)
                    for ih, wh in ((i0, 1.0 - fh), (i1, fh)):
                        for jy, wy in ((j0, 1.0 - fy_r), (j0 + 1, fy_r)):
                            for kc, wc in ((k0, 1.0 - fc_r), (k0 + 1, fc_r)):
                                delta = delta + hr_t[:, ih, jy, kc] * (wh * wy * wc)
                    w = torch.clamp((chroma - 0.03) / 0.05, 0.0, 1.0)
                    ang = torch.deg2rad(hue + delta[2] * w)
                    chroma = chroma * (1.0 + delta[1] * w)
                    ycc_out = torch.stack((luma + delta[0] * w, -chroma * torch.cos(ang), -chroma * torch.sin(ang)), dim=1)
                    out = ycc_out @ m_fw_ycc_inv.T

            # 11. Color Chrome Effect & Color Chrome FX Blue
            out = _apply_cce_t(out, ef, bf, mode_d, dev)

            # 12. Output encoding
            disp = _encode_output_t(out, out_space, dev)
            out_host[start:end] = disp.cpu().numpy()

    return out_host.reshape(H, W, 3)
