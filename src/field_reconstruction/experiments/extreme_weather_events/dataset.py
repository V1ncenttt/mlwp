from __future__ import annotations
import os
import torch
from typing import Tuple
from .config import ExtremeConfig
from .detectors import select_extreme_indices_from_Y

def build_paths(variables, percent: float, mode: str, nvars: int, root: str) -> Tuple[str, str]:
    tag = "_".join([v.replace("/", "_") for v in variables])
    base = f"{tag}_{percent}p_{mode}_{nvars}vars"
    in_path  = os.path.join(root, f"{base}_test.pt")
    out_dir  = os.path.join(root, "extremes")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_test.pt")
    return in_path, out_path

def slice_time(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx.to(x.device)) if idx.numel() > 0 else x[:0]

def create_extreme_test_dataset(
    variables,
    percent: float,
    reco_mode: str,
    data_dir: str = "../../data/weatherbench2_fieldreco/",
    cfg: ExtremeConfig = ExtremeConfig(),
) -> str:
    """
    Load *_test.pt (dict {'X':[T,Cx,H,W], 'Y':[T,5,H,W]}),
    keep only snapshots flagged as extreme (detected from Y),
    save same format to .../extremes/*_test_extremes.pt.
    Returns the output path.
    """
    in_path, out_path = build_paths(variables, percent, reco_mode, len(variables), data_dir)
    data = torch.load(in_path, map_location="cpu")  # {'X':..., 'Y':...}

    X, Y = data["X"], data["Y"]
    keep = select_extreme_indices_from_Y(Y, cfg=cfg)

    X_sel, Y_sel = slice_time(X, keep), slice_time(Y, keep)
    torch.save({"X": X_sel, "Y": Y_sel}, out_path)
    print(f"✅ Extreme test saved: {out_path}  (kept {Y_sel.shape[0]} / {Y.shape[0]} frames)")
    return out_path