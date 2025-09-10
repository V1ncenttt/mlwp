from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ExtremeConfig:
    # Heat/cold (per-grid z on T2m)
    heat_z: float = 2.5
    cold_z: float = -2.5
    heat_min_frac: float = 0.05      # ≥5% of grid hot to flag frame
    cold_min_frac: float = 0.05

    # Bands
    tc_lat_abs_max: float = 30.0
    etc_lat_abs_min: float = 25.0

    # GLOBAL (dataset-wide) percentiles for snapshot features
    q_vmag: float  = 0.995           # |V| global threshold
    q_zeta: float  = 0.999           # vorticity global threshold
    q_lap: float   = 0.999           # Laplacian(MSLP)
    q_gradT: float = 0.999           # |∇T|
    q_mqf: float   = 0.995           # TCWV*|V|

    # Area gating (as fraction of grid) – avoidsa “salt-and-pepper”
    tc_min_frac: float  = 0.002      # ~3–5 cells on your grid
    tc_max_frac: float  = 0.05       # compact systems
    etc_min_frac: float = 0.01       # ETCs are broader
    ar_min_frac: float  = 0.02
    ar_max_frac: float  = 0.25

    # batching
    batch_size: int = 64

    # Y channel order: (T2m, U10, V10, MSLP, TCWV)
    y_channel_order: Tuple[str, str, str, str, str] = ("t2m", "u10", "v10", "mslp", "tcwv")