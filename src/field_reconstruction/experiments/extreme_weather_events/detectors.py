from __future__ import annotations
import torch
from .config import ExtremeConfig
from .numerics import cdiff, laplacian, zscore_per_grid, abs_lat_grid, spatial_percentile_batched

@torch.no_grad()
def select_extreme_indices_from_Y(Y: torch.Tensor, cfg: ExtremeConfig) -> torch.Tensor:
    """
    Snapshot-only event screening on truth fields Y.
    Y: [T,5,H,W], channels = (T2m, U10, V10, MSLP, TCWV).
    Returns LongTensor of kept time indices.
    """
    assert Y.ndim == 4 and Y.size(1) == 5, "Y must be [T,5,H,W] in the fixed order."
    T, _, H, W = Y.shape
    dev, dt = Y.device, Y.dtype

    t2m, u10, v10, mslp, tcwv = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4]
    lat_abs = abs_lat_grid(H, W, dev, dt)
    zT = zscore_per_grid(t2m)

    flags = {
        "heat": torch.zeros(T, dtype=torch.bool, device=dev),
        "cold": torch.zeros(T, dtype=torch.bool, device=dev),
        "tc":   torch.zeros(T, dtype=torch.bool, device=dev),
        "etc":  torch.zeros(T, dtype=torch.bool, device=dev),
        "ar":   torch.zeros(T, dtype=torch.bool, device=dev),
    }

    B = cfg.batch_size
    for s in range(0, T, B):
        e = min(s + B, T)
        Tb, Ub, Vb, Pb = t2m[s:e], u10[s:e], v10[s:e], mslp[s:e]
        Zb, Wb = zT[s:e], tcwv[s:e]

        Vmag  = torch.sqrt(Ub**2 + Vb**2)
        zeta  = cdiff(Vb, -1) - cdiff(Ub, -2)
        lapP  = laplacian(Pb)
        dTy   = cdiff(Tb, -2).abs()
        dTx   = cdiff(Tb, -1).abs()
        gradT = torch.sqrt(dTy**2 + dTx**2)
        MQF   = Wb * Vmag

        zeta_thr = spatial_percentile_batched(zeta,  cfg.p_zeta)
        lapP_thr = spatial_percentile_batched(lapP,  cfg.p_lap)
        gT_thr   = spatial_percentile_batched(gradT, cfg.p_gradT)
        mqf_thr  = spatial_percentile_batched(MQF,   cfg.p_mqf)

        heat = (Zb >= cfg.heat_z)
        cold = (Zb <= cfg.cold_z)
        tc   = (lat_abs <= cfg.tc_lat_abs_max) & (Vmag >= cfg.gale_ms) & (zeta >= zeta_thr) & (lapP >= lapP_thr)
        etc  = (lat_abs >= cfg.etc_lat_abs_min) & (lapP >= lapP_thr) & (gradT >= gT_thr)
        ar   = (MQF >= mqf_thr)

        flags["heat"][s:e] = heat.view(e - s, -1).any(1)
        flags["cold"][s:e] = cold.view(e - s, -1).any(1)
        flags["tc"][s:e]   = tc.view(  e - s, -1).any(1)
        flags["etc"][s:e]  = etc.view( e - s, -1).any(1)
        flags["ar"][s:e]   = ar.view(  e - s, -1).any(1)

    keep = (flags["heat"] | flags["cold"] | flags["tc"] | flags["etc"] | flags["ar"]).nonzero(as_tuple=False).squeeze(-1).cpu()
    return keep