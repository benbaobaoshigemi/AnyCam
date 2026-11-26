"""Colour constants, chromatic adaptation and CIE 1960 (CCT, Duv) math shared by all engines."""
import math

import numpy as np

PROPHOTO_TO_XYZ = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
], dtype=np.float64)
XYZ_TO_PROPHOTO = np.linalg.inv(PROPHOTO_TO_XYZ)
XYZ_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
XYZ_D50 = np.array([0.9642, 1.0000, 0.8249], dtype=np.float64)

BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)
BRADFORD_INV = np.linalg.inv(BRADFORD)

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
P3_TO_XYZ = np.array([
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
], dtype=np.float64)
ADOBE_TO_XYZ = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
], dtype=np.float64)
SRGB_TO_P3 = np.linalg.inv(P3_TO_XYZ) @ SRGB_TO_XYZ
SRGB_TO_ADOBE = np.linalg.inv(ADOBE_TO_XYZ) @ SRGB_TO_XYZ
ADOBE_GAMMA = 563.0 / 256.0

# DNG CalibrationIlluminant code -> CCT
ILLUM_T = {17: 2856.0, 21: 6504.0, 19: 6774.0, 20: 5503.0, 23: 5003.0, 24: 3200.0, 18: 4874.0, 22: 7504.0}

# CIE 1960 uv Planckian locus lookup (Kim et al. cubic fit), dense in mired
_MIRED = np.linspace(1e6 / 25000.0, 1e6 / 1667.0, 20001)
_T = 1e6 / _MIRED
_PU = (0.860117757 + 1.54118254e-4 * _T + 1.28641212e-7 * _T ** 2) / (1.0 + 8.42420235e-4 * _T + 7.08145163e-7 * _T ** 2)
_PV = (0.317398726 + 4.22806245e-5 * _T + 4.20481691e-8 * _T ** 2) / (1.0 - 2.89741816e-5 * _T + 1.61456053e-7 * _T ** 2)


def cat_bradford(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    s, d = BRADFORD @ src, BRADFORD @ dst
    return BRADFORD_INV @ np.diag(d / s) @ BRADFORD


def cct_duv(xyz: np.ndarray) -> tuple[float, float]:
    """Correlated colour temperature (nearest Planckian point in CIE 1960 uv) and signed Duv."""
    v_xyz = np.asarray(xyz, dtype=np.float64)
    denom = max(1e-12, float(v_xyz[0] + 15.0 * v_xyz[1] + 3.0 * v_xyz[2]))
    u = 4.0 * float(v_xyz[0]) / denom
    v = 6.0 * float(v_xyz[1]) / denom
    i = int(np.argmin((_PU - u) ** 2 + (_PV - v) ** 2))
    return float(_T[i]), float(np.sign(v - _PV[i]) * np.hypot(u - _PU[i], v - _PV[i]))


def white_xyz_from_cct_duv(cct: float, duv: float = 0.0) -> np.ndarray:
    """Convert (CCT, Duv) in CIE 1960 uv into CIE XYZ (Y=1.0)."""
    tc = float(np.clip(cct, _T[-1], _T[0]))
    i = int(np.argmin(np.abs(_T - tc)))
    u0, v0 = float(_PU[i]), float(_PV[i])
    i_lo = max(0, i - 1)
    i_hi = min(len(_T) - 1, i + 1)
    tu = float(_PU[i_hi] - _PU[i_lo])
    tv = float(_PV[i_hi] - _PV[i_lo])
    nu, nv = -tv, tu
    n_len = max(1e-12, math.hypot(nu, nv))
    nu, nv = nu / n_len, nv / n_len
    if nv < 0:
        nu, nv = -nu, -nv
    u = u0 + duv * nu
    v = v0 + duv * nv
    d = 2.0 * u - 8.0 * v + 4.0
    x = 3.0 * u / d
    y = 2.0 * v / d
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def parse_wb_cct_duv(wb_spec: str) -> tuple[str, float, float]:
    """Parse 'camera' or '<CCT>[K][,<Duv>]' into (mode, cct, duv)."""
    s = wb_spec.strip().lower()
    if s == "camera":
        return "camera", 0.0, 0.0
    parts = [p.strip() for p in s.split(",")]
    if len(parts) > 2:
        raise ValueError(f"Invalid --wb '{wb_spec}'. Use 'camera' or '<CCT>[K][,<Duv>]'.")
    k_str = parts[0][:-1] if parts[0].endswith("k") else parts[0]
    cct = float(k_str)
    duv = float(parts[1]) if len(parts) == 2 else 0.0
    return "manual", cct, duv
