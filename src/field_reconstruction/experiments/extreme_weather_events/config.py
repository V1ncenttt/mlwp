from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ExtremeConfig:
    """Thresholds & options for snapshot-only extreme detection."""
    # heat/cold (z-score on 2m temperature, per-grid climatology)
    heat_z: float = 2.0
    cold_z: float = -2.0

    # tropical cyclone (TC) snapshot proxy
    gale_ms: float = 17.0           # |V| threshold (m/s)
    tc_lat_abs_max: float = 30.0
    p_zeta: float = 0.99            # vorticity percentile
    p_lap: float = 0.99             # Laplacian(MSLP) percentile

    # extratropical cyclone (ETC) proxy
    etc_lat_abs_min: float = 25.0
    p_gradT: float = 0.99           # |∇T| percentile

    # atmospheric river (AR) proxy (moisture flux ≈ TCWV * |V|)
    p_mqf: float = 0.97

    # batching
    batch_size: int = 64

    # Y channel order (fixed by your pipeline):
    # 0: 2m_temperature, 1: 10m_u, 2: 10m_v, 3: mslp, 4: tcwv
    y_channel_order: Tuple[str, str, str, str, str] = (
        "t2m", "u10", "v10", "mslp", "tcwv"
    )