"""Adobe DCP reader and the dual-illuminant Adobe Standard profile used by the dual-DCP bridge."""
import struct
from pathlib import Path

import numpy as np

from .color import ILLUM_T, XYZ_D50, XYZ_D65, XYZ_TO_PROPHOTO, PROPHOTO_TO_XYZ, cat_bradford, cct_duv


def read_dcp(path: Path) -> dict:
    """Read TIFF-structured Adobe .dcp tags (either byte order)."""
    b = Path(path).read_bytes()
    bo = "<" if b[:2] == b"II" else ">"
    off = struct.unpack(bo + "I", b[4:8])[0]
    n = struct.unpack(bo + "H", b[off:off + 2])[0]
    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    fmt = {1: "B", 3: "H", 4: "I", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d"}
    tags = {}
    for i in range(n):
        tag, ty, c, v = struct.unpack(bo + "HHII", b[off + 2 + i * 12:off + 14 + i * 12])
        s = size[ty] * c
        d = b[off + 10 + i * 12:off + 14 + i * 12] if s <= 4 else b[v:v + s]
        if ty == 2:
            tags[tag] = d.rstrip(b"\x00").decode("latin1")
        elif ty in (5, 10):
            tags[tag] = np.asarray(struct.unpack(f"{bo}{2 * c}{'I' if ty == 5 else 'i'}", d), np.float64)
        else:
            tags[tag] = np.asarray(struct.unpack(f"{bo}{c}{fmt[ty]}", d[:s]))
    return tags


class DcpProfile:
    """Adobe Standard profile: dual-illuminant ColorMatrix / ForwardMatrix / 2-D HueSatMap.

    Built either from a .dcp file or from explicit matrices (target cameras whose profile
    ships inside a look asset).
    """

    def __init__(self, dcp_path: Path | None = None, *, name="built-in", cm=None, fm=None, hsm=None,
                 t1=2856.0, t2=6504.0, baseline_ev=0.0, srgb_encoded=False):
        if dcp_path is not None:
            d = read_dcp(dcp_path)
            rm = lambda k: (d[k][0::2] / d[k][1::2]).reshape(3, 3)
            self.name = Path(dcp_path).stem
            self.cm = [rm(50721), rm(50722) if 50722 in d else rm(50721)]
            self.t1 = ILLUM_T.get(int(d[50778][0]), 2856.0) if 50778 in d else 2856.0
            self.t2 = ILLUM_T.get(int(d[50779][0]), 6504.0) if 50779 in d else 6504.0
            self.fm = [rm(50964), rm(50965) if 50965 in d else rm(50964)] if 50964 in d else None
            self.baseline_ev = float(d.get(51109, [0.0])[0])
            self.hsm = None
            self.hsm_srgb = int(d.get(51107, [0])[0]) == 1
            if 50937 in d and 50938 in d:
                hd, sd, vd = (int(x) for x in d[50937])
                if vd != 1:
                    raise ValueError(f"{dcp_path}: 3-D HueSatMap (V divisions={vd}) is not supported by the dual-DCP bridge")
                hsm1 = d[50938].astype(np.float64).reshape(hd, sd, 3)
                hsm2 = d[50939].astype(np.float64).reshape(hd, sd, 3) if 50939 in d else hsm1
                self.hsm = [hsm1, hsm2]
        else:
            self.name = name
            self.cm = cm
            self.fm = fm
            self.hsm = hsm
            self.t1 = float(t1)
            self.t2 = float(t2)
            self.baseline_ev = float(baseline_ev)
            self.hsm_srgb = bool(srgb_encoded)

    def weight(self, cct: float) -> float:
        if abs(self.t2 - self.t1) < 1e-6:
            return 0.5
        return float(np.clip((1.0 / cct - 1.0 / self.t1) / (1.0 / self.t2 - 1.0 / self.t1), 0.0, 1.0))

    def at(self, cct: float):
        w = self.weight(cct)
        mix = lambda a: a[0] * (1.0 - w) + a[1] * w
        cm = mix(self.cm)
        fm = mix(self.fm) if self.fm is not None else None
        hsm = mix(self.hsm) if self.hsm is not None else None
        return cm, fm, hsm

    def neutral_of_white(self, xyz: np.ndarray) -> tuple[np.ndarray, float]:
        cct = cct_duv(xyz)[0]
        n = self.at(cct)[0] @ np.asarray(xyz, dtype=np.float64)
        return n / n[1], cct

    def white_of_neutral(self, n: np.ndarray) -> tuple[np.ndarray, float]:
        cct = 5000.0
        xyz = XYZ_D65.copy()
        n_arr = np.asarray(n, dtype=np.float64)
        for _ in range(30):
            xyz = np.linalg.solve(self.at(cct)[0], n_arr)
            xyz = xyz / xyz[1]
            new = cct_duv(xyz)[0]
            if abs(1.0 / new - 1.0 / cct) < 1e-9:
                break
            cct = new
        return xyz, cct

    def to_scene_params(self, white_xyz: np.ndarray):
        """(cam -> ProPhoto D50 matrix, HueSatMap or None, hsm_srgb) under white_xyz."""
        n, cct = self.neutral_of_white(white_xyz)
        cm, fm, hsm = self.at(cct)
        if fm is not None:
            m = XYZ_TO_PROPHOTO @ fm @ np.diag(1.0 / n)
        else:
            inv_cm = np.linalg.inv(cm)
            s = 1.0 / (inv_cm @ n)[1]
            m = XYZ_TO_PROPHOTO @ cat_bradford(np.asarray(white_xyz, np.float64), XYZ_D50) @ (s * inv_cm)
        return m, hsm, self.hsm_srgb

    def from_scene_params(self, white_xyz: np.ndarray):
        """(ProPhoto D50 -> cam matrix, HueSatMap or None, hsm_srgb) under white_xyz."""
        n, cct = self.neutral_of_white(white_xyz)
        cm, fm, hsm = self.at(cct)
        if fm is not None:
            m = np.diag(n) @ np.linalg.inv(fm) @ PROPHOTO_TO_XYZ
        else:
            inv_cm = np.linalg.inv(cm)
            s = 1.0 / (inv_cm @ n)[1]
            cam_to_pp = XYZ_TO_PROPHOTO @ cat_bradford(np.asarray(white_xyz, np.float64), XYZ_D50) @ (s * inv_cm)
            m = np.linalg.inv(cam_to_pp)
        return m, hsm, self.hsm_srgb
