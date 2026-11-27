"""AnyCam command line: anycam {fuji,m9,m10r,m240p} --raw RAW --dcp DCP [options]."""
import argparse
import time
from pathlib import Path

from .core.dcp import DcpProfile
from .core.denoise import DENOISE_LEVELS, denoise_np
from .core.gpu import device_name, resolve_device
from .core.output import OUT_SPACES, save_output
from .core.raw import load_raw, orient_np


def _common(p: argparse.ArgumentParser, raw_required: bool = True) -> None:
    g = p.add_argument_group("input / output")
    g.add_argument("--raw", type=Path, required=raw_required, help="Input RAW (.DNG, .RAF, .ARW, .CR3, .NEF, ...)")
    g.add_argument("--dcp", type=Path, required=raw_required, help="Adobe Standard .dcp of the camera that shot the RAW")
    g.add_argument("--out", "-o", type=Path, help="Output image (.png, .jpg, .tif = 16-bit); default next to the RAW")
    g.add_argument("--out-space", choices=OUT_SPACES, default="srgb", help="Output colour space (default: srgb)")
    g = p.add_argument_group("processing")
    g.add_argument("--ev", type=float, default=0.0, help="Exposure compensation in EV (default: 0)")
    g.add_argument("--denoise", choices=DENOISE_LEVELS, default="off",
                   help="Noise-adaptive RAW denoise before the engine: luma bilateral + strong chroma (default: off)")
    g.add_argument("--no-hsm", action="store_true", help="Skip HueSatMap alignment (ForwardMatrix only)")
    g.add_argument("--half-size", action="store_true", help="Half-resolution demosaic for fast preview")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (default: auto = CUDA when available)")


def build_parser() -> argparse.ArgumentParser:
    from .engines.m9_awb import PRESET_NAMES

    p = argparse.ArgumentParser(
        prog="anycam",
        description="AnyCam: GPU camera colour engines (Fujifilm X100VI, Leica M9 / M10-R / M (Typ 240)) for any RAW + Adobe Standard DCP.",
    )
    sub = p.add_subparsers(dest="engine", required=True, metavar="{fuji,m9,m10r,m240p}")

    f = sub.add_parser("fuji", help="Fujifilm X100VI film simulations (20 films) + film grain")
    _common(f, raw_required=False)
    g = f.add_argument_group("Fujifilm X100VI")
    g.add_argument("--film", default="provia", help="Film simulation (default: provia); see --list-films")
    g.add_argument("--list-films", action="store_true", help="List the 20 film simulations and exit")
    g.add_argument("--dr", type=int, choices=[100, 200], default=100, help="Dynamic range DR100 / DR200 (default: 100)")
    g.add_argument("--highlight", type=float, default=0.0, help="Highlight tone -2 .. +4 (default: 0)")
    g.add_argument("--shadow", type=float, default=0.0, help="Shadow tone -2 .. +4 (default: 0)")
    g.add_argument("--color", type=int, default=0, help="Color -4 .. +4 (default: 0)")
    g.add_argument("--wb-r", type=int, default=0, help="WB shift red -9 .. +9 (default: 0)")
    g.add_argument("--wb-b", type=int, default=0, help="WB shift blue -9 .. +9 (default: 0)")
    g.add_argument("--cce", type=int, choices=[0, 1, 2], default=0, help="Color Chrome Effect: 0 off, 1 weak, 2 strong")
    g.add_argument("--fxb", type=int, choices=[0, 1, 2], default=0, help="Color Chrome FX Blue: 0 off, 1 weak, 2 strong")
    g.add_argument("--grain", choices=["off", "weak", "strong"], default="off", help="Grain effect (default: off)")
    g.add_argument("--grain-size", choices=["small", "large"], default="small", help="Grain size (default: small)")
    g.add_argument("--grain-seed", type=int, default=0, help="Grain random seed (default: 0)")
    g.add_argument("--native-grain", action="store_true", help="1:1 pixel grain instead of 48 um resolution-invariant scaling")
    g.add_argument("--cct", type=float, default=None, help="Override the frame CCT in Kelvin (default: solved from camera WB)")

    m = sub.add_parser("m9", help="Leica M9 (fw 1.216) colour, 5x5 contrast/saturation, bf547 AWB")
    _common(m)
    g = m.add_argument_group("Leica M9")
    g.add_argument("--wb", default="auto",
                   help=f"auto (M9 AWB, default), camera (RAW WB), preset ({', '.join(PRESET_NAMES)}) or Kelvin e.g. 5200K")
    g.add_argument("--contrast", type=int, choices=[-2, -1, 0, 1, 2], default=0, help="Contrast -2 .. +2 (default: 0)")
    g.add_argument("--saturation", type=int, choices=[-2, -1, 0, 1, 2], default=0, help="Saturation -2 .. +2 (default: 0)")
    g.add_argument("--mono", action="store_true", help="The camera's B&W (luminance of the camera RGB, its own five tone curves)")
    g.add_argument("--bv", type=int, default=None, help="Override metered BV (APEX BV x 256) for --wb auto")
    g.add_argument("--iso-gate", default="off", help="AWB ISO block gate: off (default, thr 100), raw (EXIF ISO) or an ISO value")
    g.add_argument("--ref", choices=["reference", "factory"], default="reference", help="M9 body calibration (default: reference)")

    r = sub.add_parser("m10r", help="Leica M10-R (fw 40.26.35.51) colour, 3x3 contrast/saturation, Monochrome")
    _common(r)
    g = r.add_argument_group("Leica M10-R")
    g.add_argument("--wb", default="camera", help="camera (RAW WB, default) or <CCT>[K][,<Duv>] e.g. 5200K or 4700K,-0.002")
    g.add_argument("--contrast", type=int, choices=[-1, 0, 1], default=0, help="Contrast -1 Low / 0 Medium / +1 High")
    g.add_argument("--saturation", type=int, choices=[-1, 0, 1], default=0, help="Saturation -1 Low / 0 Medium / +1 High")
    g.add_argument("--mono", action="store_true", help="Monochrome")

    t = sub.add_parser("m240p", help="Leica M (Typ 240) / M-P (fw 2.1.0.0) Film Modes + Monochrome")
    _common(t)
    g = t.add_argument_group("Leica M (Typ 240)")
    g.add_argument("--wb", default="camera",
                   help="camera (RAW WB, default), auto (M240 AWB) or <CCT>[K][,<Duv>] e.g. 5200K or 5200K,-0.002")
    g.add_argument("--bv", type=int, default=None, help="Override metered BV (APEX BV x 256) for --wb auto")
    g.add_argument("--mode", choices=["standard", "vivid", "natural"], default="standard", help="Film Mode (default: standard)")
    g.add_argument("--contrast", type=int, choices=[-2, -1, 0, 1, 2], default=0, help="Standard contrast -2 .. +2 (default: 0)")
    g.add_argument("--saturation", type=int, choices=[-2, -1, 0, 1, 2], default=0, help="Standard saturation -2 .. +2 (default: 0)")
    g.add_argument("--mono", action="store_true", help="BT.601 Monochrome")
    return p


def _parse_iso_gate(spec: str, raw_iso: float | None) -> float | None:
    s = spec.strip().lower()
    if s in ("off", "none", "0", "false"):
        return None
    if s == "raw":
        return raw_iso
    return float(s)


def run_fuji(args, frame, dev):
    from .engines import fuji
    from .engines.fuji_grain import apply_grain_gpu

    cam_to_pp, info = fuji.fuji_scene(frame, args.dcp, ev=args.ev, cct_override=args.cct)
    look = fuji.load_film_look(args.film)
    out = fuji.render_image_gpu(
        frame.cam_rgb, cam_to_pp, look, cct=info["cct"],
        src_hsm_table=info["hsm_table"], src_hsm_srgb=info["hsm_srgb"], use_hsm=not args.no_hsm,
        dr=args.dr, highlight=args.highlight, shadow=args.shadow, color_step=args.color,
        wb_r=args.wb_r, wb_b=args.wb_b, color_chrome=args.cce, color_chrome_blue=args.fxb,
        out_space=args.out_space, device=args.device,
    )
    out = orient_np(out, frame.flip)
    out = apply_grain_gpu(
        out, strength=args.grain, size=args.grain_size, is_acros=args.film.strip().lower().startswith("acros"),
        seed=args.grain_seed, resolution_invariant=not args.native_grain, device=args.device,
    )
    desc = (f"film={args.film} dr={args.dr} hl={args.highlight:+.1f} sh={args.shadow:+.1f} col={args.color:+d} "
            f"cce={args.cce} fxb={args.fxb} grain={args.grain}/{args.grain_size} | CCT={info['cct']:.0f}K")
    return out, desc, f"fuji_{args.film}"


def run_m9(args, frame, dev):
    from .engines import m9
    from .engines.m9_awb import REF_FACTORY, REF_REFERENCE_BODY, Awb

    src = DcpProfile(args.dcp)
    look = m9.decode_m9_bin()
    k, bv_meter = m9.exposure(frame, src, args.ev)
    bv = args.bv if args.bv is not None else bv_meter
    wb = m9.solve_wb(
        frame.cam_rgb, src, frame.neutral, k, bv, args.wb, look, Awb(),
        ref=REF_REFERENCE_BODY if args.ref == "reference" else REF_FACTORY,
        iso=_parse_iso_gate(args.iso_gate, frame.iso), use_hsm=not args.no_hsm, device=args.device,
    )
    out = m9.render(frame.cam_rgb, src, look, wb, k, contrast=args.contrast, saturation=args.saturation, mono=args.mono,
                    use_hsm=not args.no_hsm, out_space=args.out_space, device=args.device)
    out = orient_np(out, frame.flip)
    desc = (f"wb={wb['mode']} xy={wb['xy']} CCT={wb['cct']:.0f}K Duv={wb['duv']:+.4f} bv={bv} | "
            f"C={args.contrast:+d} " + ("B&W" if args.mono else f"S={args.saturation:+d}"))
    return out, desc, f"m9_{args.wb}_c{args.contrast:+d}_" + ("mono" if args.mono else f"s{args.saturation:+d}")


def run_m10r(args, frame, dev):
    from .engines import m10r

    src = DcpProfile(args.dcp)
    look = m10r.decode_m10r_bin()
    k = m10r.exposure(frame, src, args.ev)
    wb = m10r.solve_wb(src, frame.neutral, args.wb, look)
    out = m10r.render(frame.cam_rgb, src, look, wb, k, contrast=args.contrast, saturation=args.saturation,
                      mono=args.mono, use_hsm=not args.no_hsm, out_space=args.out_space, device=args.device)
    out = orient_np(out, frame.flip)
    sat = "mono" if args.mono else f"s{args.saturation:+d}"
    desc = (f"wb={wb['mode']} fw={wb['cct_fw']:.0f}K({wb['duv_fw']:+.4f}) "
            f"scene={wb['cct_scene']:.0f}K({wb['duv_scene']:+.4f}) | C={args.contrast:+d} S={sat}")
    return out, desc, f"m10r_{args.wb}_c{args.contrast:+d}_{sat}"


def run_m240p(args, frame, dev):
    from .engines import m240p

    src = DcpProfile(args.dcp)
    look = m240p.decode_m240p_bin()
    k = m240p.exposure(frame, src, args.ev)
    bv = args.bv if args.bv is not None else m240p.metered_bv(frame)
    wb = m240p.solve_wb(src, frame.neutral, args.wb, look, cam_rgb=frame.cam_rgb, k_to_m240=k, bv=bv,
                        use_hsm=not args.no_hsm, device=args.device)
    out = m240p.render(frame.cam_rgb, src, look, wb, k, mode=args.mode, contrast=args.contrast,
                       saturation=args.saturation, mono=args.mono, use_hsm=not args.no_hsm,
                       out_space=args.out_space, device=args.device)
    out = orient_np(out, frame.flip)
    style = f"{args.mode} C={args.contrast:+d} S={args.saturation:+d}" if args.mode == "standard" else args.mode
    if args.mono:
        style += " mono"
    awb = f" xy={wb['xy']} bv={bv} rounds={wb['rounds']}" if wb["mode"] == "auto" else ""
    desc = f"wb={wb['mode']}{awb} scene={wb['cct_scene']:.0f}K({wb['duv_scene']:+.4f}) | {style}"
    tag = f"m240p_{args.wb}_{args.mode}" + (f"_c{args.contrast:+d}_s{args.saturation:+d}" if args.mode == "standard" else "")
    return out, desc, tag + ("_mono" if args.mono else "")


RUNNERS = {"fuji": run_fuji, "m9": run_m9, "m10r": run_m10r, "m240p": run_m240p}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.engine == "fuji":
        from .engines.fuji import X100VI_FILMS
        if args.list_films:
            print("Film simulations:")
            for name in X100VI_FILMS:
                print(" ", name)
            return 0
        if not args.raw or not args.dcp:
            parser.error("fuji: --raw and --dcp are required (unless --list-films)")
        if args.film.strip().lower().replace("-", "_").replace(" ", "_") not in X100VI_FILMS:
            parser.error(f"fuji: unknown --film '{args.film}'; see --list-films")

    dev = resolve_device(args.device)
    t0 = time.perf_counter()
    frame = load_raw(args.raw, half_size=args.half_size)
    t_raw = time.perf_counter()
    frame.cam_rgb = denoise_np(frame.cam_rgb, args.denoise, dev)
    t_dn = time.perf_counter()
    out, desc, tag = RUNNERS[args.engine](args, frame, dev)
    t_eng = time.perf_counter()
    out_path = args.out or args.raw.with_name(f"{args.raw.stem}_{tag}.png")
    save_output(out, out_path)
    t_save = time.perf_counter()

    H, W = out.shape[:2]
    print(f"[anycam {args.engine}] {W}x{H} | {desc} | denoise={args.denoise} | space={args.out_space} | device={device_name(dev)}")
    print(f"  timing: raw={t_raw - t0:.3f}s denoise={t_dn - t_raw:.3f}s engine={t_eng - t_dn:.3f}s "
          f"save={t_save - t_eng:.3f}s total={t_save - t0:.3f}s -> {out_path}")
    return 0
