"""Same-calibrator dual-DCP bridge: source Adobe Standard -> linear ProPhoto D50 -> target Adobe Standard raw.

Both ends are Adobe Standard profiles meeting at one scene white: the source ForwardMatrix +
HueSatMap map the frame into ProPhoto D50, the target HueSatMap + ForwardMatrix are inverted
back into target camera raw. ProfileToneCurve is never used.
"""
import numpy as np
import torch

from .dcp import DcpProfile
from .gpu import apply_hsm_t, as_t, build_inv_hsm_t, inverse_hsm_t


class DualDcpBridge:
    def __init__(self, src: DcpProfile, dst: DcpProfile, white_xyz: np.ndarray, k: float,
                 use_hsm: bool, dev: torch.device, inverse: str = "table", rounds: int = 24):
        """inverse='table' uses the precomputed inverse HueSatMap (full frames);
        inverse='iterate' runs `rounds` plain fixed-point steps (M9 AWB loop)."""
        m_to_pp, src_hsm, self.src_srgb = src.to_scene_params(white_xyz)
        m_from_pp, dst_hsm, _ = dst.from_scene_params(white_xyz)
        self.m_to_pp_t = as_t(m_to_pp, dev).T.contiguous()
        self.m_from_pp_t = as_t(m_from_pp, dev).T.contiguous()
        self.src_hsm_t = as_t(src_hsm, dev) if (use_hsm and src_hsm is not None) else None
        self.dst_hsm_t = as_t(dst_hsm, dev) if (use_hsm and dst_hsm is not None) else None
        self.inv_t = build_inv_hsm_t(self.dst_hsm_t) if (self.dst_hsm_t is not None and inverse == "table") else None
        self.rounds = rounds
        self.k = float(k)

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        """Source camera RGB (N, 3) -> target camera raw (N, 3), exposure-bridged by k (not yet white balanced)."""
        pp = torch.clamp(c @ self.m_to_pp_t, min=0.0)
        if self.src_hsm_t is not None:
            pp = apply_hsm_t(pp, self.src_hsm_t, srgb_encoded=self.src_srgb)
        if self.dst_hsm_t is not None:
            if self.inv_t is not None:
                pp = inverse_hsm_t(pp, self.dst_hsm_t, inv_table=self.inv_t)
            else:
                pp = inverse_hsm_t(pp, self.dst_hsm_t, rounds=self.rounds)
        return (pp @ self.m_from_pp_t) * self.k
