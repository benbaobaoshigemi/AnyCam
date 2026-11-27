"""Leica M (Typ 240) / M-P firmware 2.1.0.0 AWB: IMG-DSP-212 0xd232c0..0xd267ff, bit exact on the camera's 30 x 30 block
statistics (MakerNote 0x0301), with the IMG_PARA + 0x518 parameter block (assets/m240p/m240_awb.bin) and the ROM default AWB_DATA.

Statistics: 30 x 30 windows of the 2000 x 2992 colour planes of the 4000 x 5984 sensor (rows 183 + 56 i, columns 128 + 94 j,
8 x 8 each); one block = the window sum of raw counts >> 4 (4 x the mean, 1.0 = white level 15000). The ISO 200 path (8 x 8
blocks: the DSP takes block >> 2) is the one run here.
"""
import math
import struct
from pathlib import Path

import numpy as np

DEFAULT_M240_AWB = Path(__file__).resolve().parent.parent.parent / "assets" / "m240p" / "m240_awb.bin"
WHITE_LEVEL = 15000.0
RAW_MAX = 16383.0

# cluster 1..14 -> BV curve column (0xd24d00)
_BV_COLUMN = {1: 0, 2: 1, 3: 0, 4: 2, 5: 3, 6: 0, 7: 4, 8: 0, 9: 6, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5}
_CENTROID = (1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14)


def _tdiv(a: int, b: int) -> int:
    """FR-V sdiv: integer division truncated toward zero."""
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _s16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def _i32(v: float) -> int:
    return int(v) if math.isfinite(v) else 0


class Awb:
    """The DSP AWB with its parameter block."""

    def __init__(self, path: Path = DEFAULT_M240_AWB):
        b = Path(path).read_bytes()
        if b[:8] != b"M240AWB1":
            raise ValueError(f"Invalid M240 AWB block in {path}")
        at = 8

        def i16(n):
            nonlocal at
            v = list(struct.unpack_from(f"<{n}h", b, at))
            at += 2 * n
            return v

        def u16(n):
            nonlocal at
            v = list(struct.unpack_from(f"<{n}H", b, at))
            at += 2 * n
            return v

        def i32(n):
            nonlocal at
            v = list(struct.unpack_from(f"<{n}i", b, at))
            at += 4 * n
            return v

        self.axis, self.pct_cfg = i16(4), i16(3)
        self.def_x, self.def_y = i16(16), i16(16)
        bv = i16(56)
        self.bv_curves = [bv[r * 7:(r + 1) * 7] for r in range(8)]
        p = i16(16)
        self.presets = [(p[2 * i], p[2 * i + 1]) for i in range(8)]
        k = i16(128)
        self.kelvin = [(k[2 * i], k[2 * i + 1]) for i in range(64)]
        self.z0, self.locus = i16(3), i16(7)
        self.thresh, self.calib = u16(6), u16(8)
        nx = i32(1)[0]
        self.x_bounds, self.ptr = i32(nx), i32(nx + 1)
        ny = i32(1)[0]
        self.y_bounds, self.zones = i32(ny), i32(ny)
        if at != len(b) or ny != self.ptr[-1]:
            raise ValueError(f"Corrupt M240 AWB block in {path}")

    # ---------------------------------------------------------------- tables
    @staticmethod
    def _search(v: int, arr, s: int, e: int) -> int:
        """0xd24970: index of the first bound above v; -1 outside."""
        if v < 0 or e <= s or v >= arr[e - 1]:
            return -1
        lo, hi = 0, e - s - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if arr[s + mid] <= v:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    def zone(self, x: int, y: int) -> int:
        ix = self._search(x, self.x_bounds, 0, len(self.x_bounds))
        if ix == -1:
            return 0
        s, e = self.ptr[ix], self.ptr[ix + 1]
        iy = self._search(y, self.y_bounds, s, e)
        return 0 if iy == -1 else self.zones[s + iy] & 0xFF

    def _bv_weight(self, bv: int, cluster: int) -> int:
        k = _BV_COLUMN[cluster]
        w0, t0, t1, t2, t3, t4, t5, w1 = (self.bv_curves[r][k] for r in range(8))
        if bv <= t0:
            return 0
        if bv < t1:
            return _s16(_tdiv((bv - t0) * w0, t1 - t0)) >> 8
        if bv <= t2:
            return _s16(w0) >> 8
        if bv <= t3:
            return _s16(w0 - (bv - t2) * _tdiv(w0, t3 - t2)) >> 8   # divides before it multiplies
        if bv <= t4:
            return 0
        if bv <= t5:
            return _s16(_tdiv((bv - t4) * w1, t5 - t4)) >> 8
        return _s16(w1) >> 8

    def _pct_weight(self, pct: int) -> int:
        hi, lo, th = self.pct_cfg
        if hi <= lo:
            return 0
        if pct <= th:
            return max(0, _s16(_tdiv(pct * lo, th)))
        if pct <= 100:
            return max(0, _s16(lo + (((pct - th) * _tdiv((hi - lo) << 8, 100 - th)) >> 8)))
        return 0

    def _ref(self):
        R0, Gr0, Gb0, B0, R1, Gr1, Gb1, B1 = self.calib
        g0, g1 = Gr0 + Gb0, Gr1 + Gb1
        return _tdiv(R0 << 13, g0), _tdiv(B0 << 13, g0), _tdiv(R1 << 13, g1), _tdiv(B1 << 13, g1)

    def gains_at(self, xc: int, yc: int) -> tuple[int, int, int]:
        """0xd26630: plane point -> Q8 gains (R, 256, B)."""
        x0, x1, y0, y1 = self.axis
        r0, b0, r1, b1 = self._ref()
        sxg = _tdiv((r0 - r1) * 4, x0 - x1) if x0 != x1 else 0
        syg = _tdiv((b0 - b1) * 4, y0 - y1) if y0 != y1 else 0
        rx = ((xc - x1) * sxg + r1 * 4) >> 6
        by = ((yc - y1) * syg + b1 * 4) >> 6
        return (0 if rx <= 0 else min(2000, 0x10000 // rx), 256, 0 if by <= 0 else min(2000, 0x10000 // by))

    def preset(self, index: int) -> tuple[int, int]:
        return self.presets[max(0, min(7, int(index)))]

    # ---------------------------------------------------------------- auto
    def run(self, blocks: np.ndarray, bv: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
        """blocks: (900, 4) or (30, 30, 4) (R, Gr, Gb, B) statistics. Returns the Q8 gains and the clamped plane point."""
        blk = np.asarray(blocks, dtype=np.int64).reshape(900, 4)
        x0, x1, y0, y1 = self.axis
        lo_th, hi_th = self.thresh[0], self.thresh[1]
        r0, b0, r1, b1 = self._ref()
        sx = _tdiv((x0 - x1) << 12, r0 - r1) if r0 != r1 else 0
        sy = _tdiv((y0 - y1) << 12, b0 - b1) if b0 != b1 else 0

        acc_x, acc_y, acc_n = [0] * 16, [0] * 16, [0] * 16
        for row in blk:
            v = [max(1, ((int(c) << 2) & 0xFFFFFFFF) >> 4) for c in row]
            if min(v) < lo_th or max(v) > hi_th:
                continue
            g = v[1] + v[2] or 10
            x = _s16((((_tdiv(v[0] << 13, g) - r1) * sx) >> 12) + x1)
            y = _s16((((_tdiv(v[3] << 13, g) - b1) * sy) >> 12) + y1)
            z = self.zone(x, y)
            if 1 <= z <= 15:
                idx = z
            elif z == 0 or 16 <= z <= 19:
                idx = 0
            else:
                continue
            acc_x[idx] += x
            acc_y[idx] += y
            acc_n[idx] += 1

        mean_x, mean_y = [0] * 16, [0] * 16
        for k in range(16):
            n, dx, dy = acc_n[k], self.def_x[k], self.def_y[k]
            if n < 10 or k in (2, 5):
                mean_x[k], mean_y[k] = dx, dy
            elif k == 0:
                mean_x[k], mean_y[k] = _tdiv(acc_x[k], n), _tdiv(acc_y[k], n)
            else:
                mean_x[k] = _tdiv(3 * dx + _tdiv(acc_x[k], n), 4)
                mean_y[k] = _tdiv(3 * dy + _tdiv(acc_y[k], n), 4)
        tot = sum(acc_n)
        pct = [_tdiv(acc_n[k] * 100, tot) if tot > 0 else 0 for k in range(16)]
        pw = {k: self._pct_weight(pct[k]) * (2 if k in (4, 5) else 1) for k in range(1, 16)}
        bw = {k: self._bv_weight(bv, k) for k in (1, 2, 4, 5, 7, 9, 11)}

        # 0xd25860: suppression of the daylight / shade clusters and small clusters, in the DSP's float32 / float64 precision
        f32 = np.float32
        p123 = pct[1] + pct[2] + pct[3]
        inh1 = inh2 = inh7 = inh9 = 0
        inh_c = [0] * 5
        if 0 < p123 <= 10:
            div = f32(f32(1.0) / f32(f32(0.1) * f32(p123)))
            inh1 = _i32(float(f32(-bw[1]) * div) * 0.9)
            inh2 = _i32(float(f32(-bw[2]) * div) * 0.9)
        if p123 > 30:
            inh7, inh9 = -(bw[7] + pw[7]), -(bw[9] + pw[9])
            inh_c = [-(bw[11] + pw[10 + i]) for i in range(5)]
        elif p123 >= 20:
            scale = float(f32(f32(p123 - 10) * f32(0.05)))
            inh7 = _i32(-(bw[7] + pw[7]) * scale)
            inh9 = _i32(-(bw[9] + pw[9]) * scale)
            inh_c = [_i32(-(bw[11] + pw[10 + i]) * scale) for i in range(5)]
        else:
            if 0 < pct[7] < 8:
                inh7 = _i32(float(f32(-bw[7]) * f32(f32(1.0) / f32(f32(0.125) * f32(pct[7])))))
            if 0 < pct[9] < 8:
                inh9 = _i32(float(f32(-bw[9]) * f32(f32(1.0) / f32(f32(0.125) * f32(pct[9])))))
            inv15 = f32(1.0 / 15.0)
            for i in range(5):
                if 0 < pct[10 + i] < 15:
                    inh_c[i] = _i32(float(f32(-bw[11]) * f32(f32(1.0) / f32(inv15 * f32(pct[10 + i])))))
        ws = [max(0, bw[1] + pw[1] + inh1), max(0, bw[2] + pw[2] + inh2), max(0, bw[4] + pw[4]), max(0, bw[5] + pw[5]),
              max(0, bw[7] + pw[7] + inh7), max(0, bw[9] + pw[9] + inh9)]
        ws += [max(0, bw[11] + pw[10 + i] + inh_c[i]) for i in range(5)]
        sw = sum(ws)
        if tot == 0:
            xa, ya = self.def_x[1], self.def_y[1]
        elif sw <= 2:
            xa, ya = _tdiv(mean_x[0] + self.def_x[1], 2), _tdiv(mean_y[0] + self.def_y[1], 2)
        else:
            xa = _tdiv(sum(w * mean_x[c] for w, c in zip(ws, _CENTROID)), sw)
            ya = _tdiv(sum(w * mean_y[c] for w, c in zip(ws, _CENTROID)), sw)

        # 0xd26240: the grey-point share; 0xd26420: the locus clamp
        pct0 = _tdiv(acc_n[0] * 100, acc_n[0] + tot) if acc_n[0] > 0 else 0
        w0max, zlo, zhi = self.z0
        w0 = 0 if pct0 < zlo else w0max if pct0 > zhi else ((pct0 - zlo) * _tdiv(w0max << 8, zhi - zlo)) >> 8
        xf = _tdiv(xa * (100 - w0) + mean_x[0] * w0, 100)
        yf = _tdiv(ya * (100 - w0) + mean_y[0] * w0, 100)
        pA, pB, pC, pD, pE, pF = self.locus[:6]
        xc, yc = xf, yf
        if xf <= pC and yf <= pD:
            xc = xf + 60 if (pC - xf) > 60 else pC
            yc = yf + 30 if (pD - yf) > 30 else pD
        elif xf <= pC and yf > pD:
            dx, dy = pA - pC, pB - pD
            slope = _tdiv(dy << 6, dx) if dx else 0
            if not (((((xf - pC) * slope) >> 6) + pD) < yf):
                xc = 0 if dy == 0 else _tdiv((yf - pD) * dx, dy) + pC
        elif xf > pC and yf <= pD:
            dx, dy = pE - pC, pF - pD
            slope = _tdiv(dy << 6, dx) if dx else 0
            y_lim = (((xf - pC) * slope) >> 6) + pD
            if not (y_lim < yf):
                yc = y_lim
        return self.gains_at(xc, yc), (xc, yc)


def sample_windows(H: int, W: int):
    """The firmware's 30 x 30 window grid scaled to an image of H x W: window rows, columns and size."""
    sy, sx = H / 2000.0, W / 2992.0
    wy, wx = max(1, int(round(8 * sy))), max(1, int(round(8 * sx)))
    ys = [max(0, min(H - wy, int(round((183 + 56 * i) * sy)))) for i in range(30)]
    xs = [max(0, min(W - wx, int(round((128 + 94 * j) * sx)))) for j in range(30)]
    return ys, xs, wy, wx


def blocks_of_means(m240_means: np.ndarray) -> np.ndarray:
    """(900, 3) window means of M240 raw (1.0 = white level) -> (900, 4) statistics (window sum of counts >> 4)."""
    counts = np.floor(np.clip(np.asarray(m240_means, np.float64) * WHITE_LEVEL, 0.0, RAW_MAX))
    v = (64 * counts).astype(np.int64) >> 4
    return np.stack([v[:, 0], v[:, 1], v[:, 1], v[:, 2]], axis=1)
