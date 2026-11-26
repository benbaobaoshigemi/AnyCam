"""RAW decoding: linear sensor-oriented demosaic, exposure baseline and EXIF metering metadata."""
import io
import math
import struct
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy
from threadpoolctl import threadpool_limits

# Fujifilm RAF: X100VI front-end input scale. The Fuji engine's kind-13 look carries it inside
# its own front stage, so a RAF feeds the Fuji engine without it; every other engine works in
# scene-referred 18% grey and folds it into the RAF baseline (DR100 ~+0.451 EV instead of +1.263 EV).
RAF_INPUT_SCALE = 0.569589


@dataclass
class RawFrame:
    cam_rgb: np.ndarray          # (H, W, 3) float32 linear camera RGB, sensor orientation, [0, 1] of white level
    wb: np.ndarray               # camera white balance multipliers (R, G, B)
    neutral: np.ndarray          # camera neutral (G / wb)
    flip: int                    # LibRaw flip code applied by orient_np before export
    raf_dr_ev: float | None      # Fuji DR100/200/400 under-exposure in EV (RAF only)
    dng_baseline_ev: float       # DNG BaselineExposure (0 when absent)
    ev100: float | None
    iso: float | None

    def raw_base_ev(self, *, raf_input_scale: bool) -> float:
        """RAW intrinsic exposure offset that maps 18% grey to the engine's scene grey.

        raf_input_scale=True folds log2(RAF_INPUT_SCALE) into RAF files (Leica engines);
        False keeps the bare Fuji headroom (Fuji engine, which applies the scale itself).
        """
        if self.raf_dr_ev is None:
            return self.dng_baseline_ev
        base = float(np.log2(15360.0 / 3200.0) - 1.0 + self.raf_dr_ev)
        if raf_input_scale:
            base = float(np.log2(15360.0 / 3200.0) - 1.0 + np.log2(RAF_INPUT_SCALE) + self.raf_dr_ev)
        return base


def _raf_dr_ev(raw_obj) -> float:
    dr_ev = 0.0
    try:
        thumb = raw_obj.extract_thumb().data
        idx = thumb.find(b"FUJIFILM")
        if idx >= 0:
            n_mn = struct.unpack_from("<H", thumb, idx + 12)[0]
            for i in range(n_mn):
                tag, _, _, val = struct.unpack_from("<HHII", thumb, idx + 14 + i * 12)
                if tag == 0x1403 and val in (100, 200, 400):
                    dr_ev = float(np.log2(val / 100.0))
                    break
    except Exception:
        pass
    return dr_ev


def _read_metadata(raw_path: Path, raw_obj) -> tuple[float, float | None, float | None]:
    """(DNG BaselineExposure, EV100, ISO) from TIFF/EXIF tags, falling back to the embedded JPEG EXIF."""
    base_ev = 0.0
    ev100 = None
    iso_val = None
    rat = lambda v: float(v[0] / v[1]) if isinstance(v, tuple) else float(v)
    try:
        import tifffile
        with tifffile.TiffFile(str(raw_path)) as tf:
            t = tf.pages[0].tags
            if 50730 in t:
                base_ev = rat(t[50730].value)
            ex = t["ExifTag"].value if "ExifTag" in t else {}
            if "ISOSpeedRatings" in ex:
                iso_raw = ex["ISOSpeedRatings"]
                iso_val = float(iso_raw[0] if isinstance(iso_raw, (tuple, list)) else iso_raw)
            if "FNumber" in ex and "ExposureTime" in ex and iso_val:
                ev100 = math.log2(rat(ex["FNumber"]) ** 2 / rat(ex["ExposureTime"])) - math.log2(iso_val / 100.0)
    except Exception:
        pass

    if ev100 is None or iso_val is None:
        try:
            from PIL import Image
            thumb = raw_obj.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                exif = Image.open(io.BytesIO(thumb.data)).getexif().get_ifd(0x8769)
                if iso_val is None and 34855 in exif:
                    iv = exif[34855]
                    iso_val = float(iv[0] if isinstance(iv, (tuple, list)) else iv)
                if ev100 is None and 33437 in exif and 33434 in exif and iso_val:
                    fn, et = float(exif[33437]), float(exif[33434])
                    ev100 = math.log2((fn ** 2) / et) - math.log2(iso_val / 100.0)
        except Exception:
            pass
    return base_ev, ev100, iso_val


def load_raw(raw_path: Path, half_size: bool = False) -> RawFrame:
    """Linear demosaic in sensor orientation; never uses LibRaw or RAW-embedded colour matrices."""
    raw_path = Path(raw_path)
    with rawpy.imread(str(raw_path)) as raw:
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
        neutral = float(wb[1]) / np.maximum(wb, 1e-9)
        flip = int(raw.sizes.flip)
        dng_base_ev, ev100, iso_val = _read_metadata(raw_path, raw)
        is_raf = raw_path.suffix.lower() == ".raf"
        raf_dr = _raf_dr_ev(raw) if is_raf else None
        # LibRaw's OpenMP X-Trans interpolation races (repeat decodes of one RAF differ on ~0.03 % of
        # pixels at full size, ~12 % at half size); X-Trans decodes run on one thread, Bayer and
        # linear DNG stay multi-threaded (they are deterministic).
        xtrans = raw.raw_pattern is not None and raw.raw_pattern.shape == (6, 6)
        with threadpool_limits(limits=1, user_api="openmp") if xtrans else nullcontext():
            u16 = raw.postprocess(
                half_size=half_size,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,
                user_wb=[1, 1, 1, 1],
                output_color=rawpy.ColorSpace.raw,
                adjust_maximum_thr=0.0,
                user_flip=0,
            )
    cam_rgb = u16.astype(np.float32) / 65535.0
    return RawFrame(cam_rgb, wb, neutral, flip, raf_dr, 0.0 if is_raf else dng_base_ev, ev100, iso_val)


def orient_np(img: np.ndarray, flip: int) -> np.ndarray:
    """Apply LibRaw flip orientation (same result as rawpy's own user_flip=None output); returns a contiguous array."""
    if flip == 3:
        return np.ascontiguousarray(np.rot90(img, 2, (0, 1)))
    if flip in (5, 8):
        return np.ascontiguousarray(np.rot90(img, 1, (0, 1)))
    if flip == 6:
        return np.ascontiguousarray(np.rot90(img, 3, (0, 1)))
    return img
