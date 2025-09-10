import torch
from .config import ExtremeConfig
from .numerics import cdiff, laplacian, zscore_per_grid, abs_lat_grid, global_quantile_hist

def _area_frac(mask: torch.Tensor) -> torch.Tensor:
    # mask: [B,H,W] bool -> [B] fraction
    B, H, W = mask.shape
    return mask.view(B, -1).float().mean(dim=1)

@torch.no_grad()
def select_extreme_indices_from_Y(Y: torch.Tensor, cfg: ExtremeConfig) -> torch.Tensor:
    """
    Snapshot-only screening with GLOBAL thresholds + area gating.
    Y: [T,5,H,W], channels = (T2m, U10, V10, MSLP, TCWV).
    Returns LongTensor of kept time indices.
    """
    assert Y.ndim == 4 and Y.size(1) == 5, "Y must be [T,5,H,W] in the fixed order."
    T, _, H, W = Y.shape
    dev, dt = Y.device, Y.dtype

    T2m, U10, V10, MSLP, TCWV = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4]
    latabs = abs_lat_grid(H, W, dev, dt)

    # -------- One global pass: thresholds computed on the WHOLE test set --------
    Vmag_all  = torch.sqrt(U10**2 + V10**2)            # [T,H,W]
    ZETA_all  = cdiff(V10, -1) - cdiff(U10, -2)        # [T,H,W]
    LAP_all   = laplacian(MSLP)                        # [T,H,W]
    dTy_all   = cdiff(T2m, -2).abs()
    dTx_all   = cdiff(T2m, -1).abs()
    GRADT_all = torch.sqrt(dTy_all**2 + dTx_all**2)    # [T,H,W]
    MQF_all   = TCWV * Vmag_all                        # [T,H,W]
    zT_all    = zscore_per_grid(T2m)                   # [T,H,W] (per-grid σ over time)

    # Flatten over all frames & space for global quantiles
    flat = lambda x: x.reshape(-1)


    vmag_thr = global_quantile_hist(Vmag_all,  cfg.q_vmag,  transform=None,  bins=4096)
    zeta_thr = global_quantile_hist(ZETA_all,  cfg.q_zeta,  transform="abs", bins=4096)
    lap_thr  = global_quantile_hist(LAP_all,   cfg.q_lap,   transform="abs", bins=4096)
    gT_thr   = global_quantile_hist(GRADT_all, cfg.q_gradT, transform=None,  bins=4096)
    mqf_thr  = global_quantile_hist(MQF_all,   cfg.q_mqf,   transform=None,  bins=4096)

    # -------- Frame-by-frame decisions with global thresholds --------
    keep_flags = torch.zeros(T, dtype=torch.bool, device=dev)

    B = cfg.batch_size
    for s in range(0, T, B):
        e = min(s + B, T)

        T2 = T2m[s:e]
        U  = U10[s:e]
        V  = V10[s:e]
        P  = MSLP[s:e]
        WV = TCWV[s:e]

        Vmag  = torch.sqrt(U**2 + V**2)
        ZETA  = cdiff(V, -1) - cdiff(U, -2)
        LAP   = laplacian(P)
        dTy   = cdiff(T2, -2).abs()
        dTx   = cdiff(T2, -1).abs()
        GRADT = torch.sqrt(dTy**2 + dTx**2)
        MQF   = WV * Vmag
        ZT    = zscore_per_grid(T2)  # per-grid σ within this batch slice (ok, stats are stable)

        Bsz = e - s
        # Heat/cold: require that a noticeable FRACTION of grid is extreme
        heat_mask = (ZT >= cfg.heat_z)
        cold_mask = (ZT <= cfg.cold_z)
        heat_ok = _area_frac(heat_mask) >= cfg.heat_min_frac
        cold_ok = _area_frac(cold_mask) >= cfg.cold_min_frac

        # TC proxy: compact blobs in tropics with strong |ζ|, |∇²p|, |V|
        tc_core = (latabs <= cfg.tc_lat_abs_max) & \
                  (Vmag >= vmag_thr) & (ZETA.abs() >= zeta_thr) & (LAP.abs() >= lap_thr)
        tc_frac = _area_frac(tc_core)
        tc_ok = (tc_frac >= cfg.tc_min_frac) & (tc_frac <= cfg.tc_max_frac)

        # ETC proxy: broad baroclinic zone outside tropics with strong |∇²p| and |∇T|
        etc_core = (latabs >= cfg.etc_lat_abs_min) & \
                   (LAP.abs() >= lap_thr) & (GRADT >= gT_thr)
        etc_frac = _area_frac(etc_core)
        etc_ok = (etc_frac >= cfg.etc_min_frac)

        # AR proxy: intense moisture flux with sizable but not global footprint
        ar_core = (MQF >= mqf_thr)
        ar_frac = _area_frac(ar_core)
        ar_ok = (ar_frac >= cfg.ar_min_frac) & (ar_frac <= cfg.ar_max_frac)

        keep_flags[s:e] = heat_ok | cold_ok | tc_ok | etc_ok | ar_ok

    keep = keep_flags.nonzero(as_tuple=False).squeeze(-1).cpu()
    return keep