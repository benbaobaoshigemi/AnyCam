"""Leica M9 firmware 1.216 auto white balance (bf547), presets, 63-step Kelvin table, and sparse 4x4 block sampler."""
import struct
from pathlib import Path
import numpy as np

DEFAULT_AWB_P3 = Path(__file__).resolve().parent.parent.parent / "assets" / "m9" / "awb_p3.bin"

# abd08 factory default calibration; reference body calibration matches main M9 sample bodies and Adobe M9 sensor model.
REF_FACTORY = (1857, 3353, 4152, 4171, 1550, 1306, 2453, 2492)
REF_REFERENCE_BODY = (2110, 3858, 4596, 4621, 1719, 1492, 2644, 2684)

# aba68 jump table 0xf10a4: preset index -> (x offset, y offset) in P3
PRESET_OFFSETS = {
    0: (0x1F2, 0x200),
    1: (0x1F8, 0x206),
    2: (0x1F6, 0x204),
    3: (0x1EC, 0x1FA),
    4: (0x1EE, 0x1FC),
    5: (0x1F0, 0x1FE),
    6: (0x1F4, 0x202),
}
PRESET_NAMES = {
    "daylight": 0,
    "cloudy": 1,
    "shade": 2,
    "tungsten": 3,
    "fluorescent": 4,
    "flash": 5,
    "preset6": 6,
}

# BF547 menu Kelvin ladder (0xd3bd8, 63 steps: 2000..5000 step 100, 5200..8000 step 200, 8300..13100 step 300)
KELVIN_STEPS = np.array(
    list(range(2000, 5001, 100)) + list(range(5200, 8001, 200)) + list(range(8300, 13101, 300)),
    dtype=np.float64,
)


def _tdiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def iso_thresh(iso: float | None) -> int:
    """a9af4 ISO-tiered minimum valid block threshold: <=640 -> 100, 800..1600 -> 300, >=2000 -> 500."""
    if iso is None or iso <= 640:
        return 100
    if iso <= 1600:
        return 300
    return 500


def sample_windows(H: int, W: int):
    """Sparse 20x30 lattice of 4x4-quad M9 FPGA windows (i=1..19, j=1..29) scaled to sensor shape (H, W)."""
    sy = H / 1736.0
    sx = W / 2608.0
    wy = max(1, round(4 * sy))
    wx = max(1, round(4 * sx))
    ys = [max(0, min(H - wy, round((-17 + 87 * i) * sy))) for i in range(1, 20)]
    xs = [max(0, min(W - wx, round((-15 + 87 * j) * sx))) for j in range(1, 30)]
    xbs = [max(0, min(W - wx, round((-16 + 87 * j) * sx))) for j in range(1, 30)]
    return ys, xs, xbs, wy, wx


def blocks_sparse_4x4(rgb_14: np.ndarray) -> np.ndarray:
    """Compute 600 M9 AWB blocks (20x30x3, 14-bit) from sensor-oriented linear M9 RGB in [0, 16383]."""
    H, W, _ = rgb_14.shape
    ys, xs, xbs, wy, wx = sample_windows(H, W)
    out = np.zeros((20, 30, 3), dtype=np.float64)
    for ii, y0 in enumerate(ys, start=1):
        for jj, (x0, xb0) in enumerate(zip(xs, xbs), start=1):
            out[ii, jj, 0] = rgb_14[y0:y0 + wy, x0:x0 + wx, 0].mean()
            out[ii, jj, 1] = rgb_14[y0:y0 + wy, x0:x0 + wx, 1].mean()
            out[ii, jj, 2] = rgb_14[y0:y0 + wy, xb0:xb0 + wx, 2].mean()
    return out.reshape(-1, 3)


class Awb:
    """Bit-exact Leica M9 bf547 firmware 1.216 AWB solver."""

    def __init__(self, p3: bytes | None = None, p3_path: Path = DEFAULT_AWB_P3):
        b = p3 if p3 is not None else Path(p3_path).read_bytes()
        self.p3 = b
        self.x_hi, self.x_lo, self.y_hi, self.y_lo = struct.unpack_from("<4h", b, 2)
        self.cx = np.array(struct.unpack_from("<14h", b, 0x10), np.int64)
        self.cy = np.array(struct.unpack_from("<14h", b, 0x38), np.int64)
        n_x = struct.unpack_from("<I", b, 0x208)[0]
        n_y = struct.unpack_from("<I", b, 0x2D4)[0]
        self.row_start = list(struct.unpack_from(f"<{n_x}i", b, 0x20C)) + [n_y]
        self.x_brk = np.array(struct.unpack_from(f"<{n_x}i", b, 0x2D8))
        self.y_brk = np.array(struct.unpack_from(f"<{n_y}i", b, 0x3A0))
        self.zone_id = np.array(struct.unpack_from(f"<{n_y}i", b, 0x8B4))
        self.kelvin_x = np.array(struct.unpack_from("<63h", b, 0xD6), dtype=np.float64)
        self.kelvin_y = np.array(struct.unpack_from("<63h", b, 0x156), dtype=np.float64)
        self.last: dict = {}

    def _ref_q13(self, ref):
        g0, g1 = ref[2] + ref[3], ref[6] + ref[7]
        return (ref[0] << 13) // g0, (ref[1] << 13) // g0, (ref[4] << 13) // g1, (ref[5] << 13) // g1

    def gain_at(self, x: float, y: float, ref=REF_REFERENCE_BODY) -> np.ndarray:
        """ab588: raw gains (R, 1, B) that neutralise plane point (x, y) on a body."""
        r0, b0, r1, b1 = self._ref_q13(ref)
        rq = r1 + (x - self.x_lo) * (r0 - r1) / (self.x_hi - self.x_lo)
        bq = b1 + (y - self.y_lo) * (b0 - b1) / (self.y_hi - self.y_lo)
        return np.array([4096.0 / rq, 1.0, 4096.0 / bq], dtype=np.float64)

    def plane_of_gain(self, g, ref=REF_REFERENCE_BODY) -> tuple[float, float]:
        r0, b0, r1, b1 = self._ref_q13(ref)
        rq, bq = 4096.0 / g[0] * g[1], 4096.0 / g[2] * g[1]
        return (
            self.x_lo + (rq - r1) * (self.x_hi - self.x_lo) / (r0 - r1),
            self.y_lo + (bq - b1) * (self.y_hi - self.y_lo) / (b0 - b1),
        )

    def preset(self, index_or_name: int | str) -> tuple[int, int]:
        idx = PRESET_NAMES[index_or_name.lower()] if isinstance(index_or_name, str) else int(index_or_name)
        ox, oy = PRESET_OFFSETS[idx]
        return struct.unpack_from("<h", self.p3, ox)[0], struct.unpack_from("<h", self.p3, oy)[0]

    def kelvin(self, k: float) -> tuple[int, int]:
        """Nearest or linearly interpolated plane point from the 63-step firmware Kelvin table."""
        kc = float(np.clip(k, KELVIN_STEPS[0], KELVIN_STEPS[-1]))
        x = float(np.interp(kc, KELVIN_STEPS, self.kelvin_x))
        y = float(np.interp(kc, KELVIN_STEPS, self.kelvin_y))
        return int(round(x)), int(round(y))

    @staticmethod
    def _a9dc0(val: int, arr: np.ndarray) -> int:
        n = len(arr)
        if val < 0 or n <= 0 or val >= int(arr[n - 1]):
            return -1
        lo, hi = 0, n
        while lo <= hi:
            mid = (lo + hi) >> 1
            if val < int(arr[mid]):
                hi = mid - 1
            else:
                lo = mid + 1
        return lo

    def zone(self, x: int, y: int) -> int:
        ix = self._a9dc0(x, self.x_brk)
        if ix == -1:
            return 0
        s, e = self.row_start[ix], self.row_start[ix + 1]
        iy = self._a9dc0(y, self.y_brk[s:e])
        return 0 if iy == -1 else int(self.zone_id[s + iy])

    def _aa03c(self, bv: int, zone: int) -> int:
        p2 = {1: 0, 3: 1, 6: 2, 7: 3, 5: 3, 8: 4, 9: 5, 10: 6}.get(zone, 6)
        R5, R2, R3, R7, I0, P0, R4, R6 = [struct.unpack_from("<h", self.p3, 0x60 + row * 14 + p2 * 2)[0] for row in range(8)]
        if not R2 < bv:
            return 0
        if bv <= R3:
            return ((bv - R2) * _tdiv(R5, R3 - R2)) >> 8
        if not R7 < bv:
            return R5 >> 8
        if not I0 < bv:
            return (R5 - (bv - R7) * _tdiv(R5, I0 - R7)) >> 8
        if not P0 < bv:
            return 0
        if R4 < bv:
            return R6 >> 8
        return (R6 - (bv - P0) * _tdiv(R6, R4 - P0)) >> 8

    def _aa768(self, pct: int) -> int:
        ymax, ymin, k = struct.unpack_from("<3h", self.p3, 0x0A)
        if ymax <= ymin or k <= 0 or k > 99 or pct <= 0 or pct > 100:
            return 0
        if pct <= k:
            return (pct * ymin) // k
        return ymin + (((((ymax - ymin) << 8) // (100 - k)) * (pct - k)) >> 8)

    @staticmethod
    def _aa974(w: dict[int, int]) -> dict[int, int]:
        z1, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13 = (w[k] for k in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
        return {
            7: z4 - z13 - ((z10 + z9 + z1 + z3 + z11) >> 1) - z12,
            3: -(z4 >> 1) - (z8 << 1) - z7 + z12 - 3 * (z9 + z10 + z11),
            8: (z6 >> 1) + 2 * z4 - z1 - z3 + 2 * z11 - z12 - z13 + (z7 >> 2),
            6: -z10,
            1: -z4 - ((z8 + z10) << 1) - (z9 << 2) + z13 - z7 - 3 * z11,
            10: -z4 - (z6 >> 1) - z7 - z1 - z12 - z13 - 2 * (z3 + z11),
            9: -z4 - 5 * (z1 + z3 + z12 + z13),
            5: (z3 >> 1) + (z7 >> 1) + (z12 >> 1) - z8 - 4 * (z9 + z10 + z11 + z13),
        }

    def run(self, blocks: np.ndarray, bv: int, ref=REF_REFERENCE_BODY, iso: float | None = None):
        """blocks: 600 x (R, G, B) raw 14-bit block means. bv: APEX BV x 256. Returns (gain, (ox, oy))."""
        r0, b0, r1, b1 = self._ref_q13(ref)
        sx = _tdiv((self.x_hi - self.x_lo) << 12, r0 - r1)
        sy = _tdiv((self.y_hi - self.y_lo) << 12, b0 - b1)
        blocks = np.asarray(blocks)
        lo_thr = iso_thresh(iso)
        valid = (blocks.min(1) > lo_thr) & (blocks.max(1) <= 13499)
        if valid.sum() < 10:
            valid = (blocks.min(1) > 1) & (blocks.max(1) < 16300)
        v = blocks[valid]
        if len(v) == 0:
            ox, oy = self.preset(0)
            self.last = dict(cnt=np.zeros(14, np.int64), W={}, w={}, fw={}, fxy=(ox, oy), bxy=(ox, oy), pct0=0, bw=0, nvalid=0)
            return self.gain_at(ox, oy, ref), (ox, oy)

        if blocks.dtype.kind in "ui":
            rq = [(int(r) << 13) // (2 * max(1, int(g))) for r, g, _ in v]
            bq = [(int(b) << 13) // (2 * max(1, int(g))) for _, g, b in v]
        else:
            rq = np.floor(v[:, 0] / np.maximum(v[:, 1], 1e-6) * 4096.0).astype(np.int64).tolist()
            bq = np.floor(v[:, 2] / np.maximum(v[:, 1], 1e-6) * 4096.0).astype(np.int64).tolist()
        xs = np.array([self.x_lo + (((q - r1) * sx) >> 12) for q in rq], np.int64)
        ys = np.array([self.y_lo + (((q - b1) * sy) >> 12) for q in bq], np.int64)
        zids = np.array([self.zone(int(x), int(y)) for x, y in zip(xs, ys)], np.int64)

        mx, my = self.cx.copy(), self.cy.copy()
        cnt = np.zeros(14, np.int64)
        for z in range(14):
            m = zids == z
            c = int(m.sum())
            cnt[z] = c
            if z == 9 or c == 0:
                continue
            xm, ym = _tdiv(int(xs[m].sum()), c), _tdiv(int(ys[m].sum()), c)
            if c <= 10:
                mx[z] = _tdiv((10 - c) * int(mx[z]) + c * xm, 10)
                my[z] = _tdiv((10 - c) * int(my[z]) + c * ym, 10)
            else:
                mx[z], my[z] = xm, ym

        zs = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        tot = int(sum(cnt[z] for z in zs))
        pct = {z: (int(cnt[z]) * 100) // tot if tot else 0 for z in zs}
        fw = {z: self._aa03c(bv, z) for z in (1, 3, 6, 7, 8, 9, 10, 5)}
        w = {z: self._aa768(pct[z]) for z in zs}
        inh = self._aa974(w)

        def comb(wz: int, ih: int, fz: int) -> int:
            s = wz + ih
            return max(0, s + (fz >> 1) if (s <= 0 and fz > 0) else s + fz)

        W = {z: comb(w[z], inh[z], fw[z]) for z in (7, 3, 9, 8, 6, 1, 10, 5)}
        ws = sum(W.values())
        fx = _tdiv(sum(W[z] * int(mx[z]) for z in W), ws) if ws else int(mx[6])
        fy = _tdiv(sum(W[z] * int(my[z]) for z in W), ws) if ws else int(my[6])

        cnt0 = int(cnt[0])
        pct0 = (100 * cnt0) // (tot + cnt0) if tot + cnt0 else 100
        lo_t, hi_t = struct.unpack_from("<2h", self.p3, 0x1D8)
        max_w = struct.unpack_from("<h", self.p3, 0x1D6)[0]
        bw = 0 if pct0 < lo_t else (((pct0 - lo_t) * ((max_w << 8) // max(1, hi_t - lo_t))) >> 8 if pct0 < hi_t else max_w)
        if cnt0:
            bx = max(fx - 90, min(fx + 90, _tdiv((100 - bw) * fx + bw * int(mx[0]), 100)))
            by = max(fy - 50, min(fy + 50, _tdiv((100 - bw) * fy + bw * int(my[0]), 100)))
        else:
            bx, by = fx, fy

        ox, oy = bx, by
        xl, yl, xmid, ymid, xr, yr = struct.unpack_from("<6h", self.p3, 0x1DC)
        if ox <= xmid:
            if oy <= ymid:
                ox, oy = xmid, ymid
            elif xmid - xl:
                y_lim = ymid + ((((yl - ymid) << 6) // (xl - xmid) * (ox - xmid)) >> 6)
                if oy >= y_lim and yl - ymid:
                    ox = xmid + _tdiv((oy - ymid) * (xl - xmid), yl - ymid)
        elif oy <= ymid and xr - xmid:
            y_lim = ymid + ((((yr - ymid) << 6) // (xr - xmid) * (ox - xmid)) >> 6)
            if oy <= y_lim:
                oy = y_lim

        self.last = dict(cnt=cnt, W=W, w=w, fw=fw, mx=mx, my=my, fxy=(fx, fy), bxy=(bx, by), pct0=pct0, bw=bw, nvalid=int(valid.sum()))
        return self.gain_at(ox, oy, ref), (ox, oy)
