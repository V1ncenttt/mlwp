from __future__ import annotations
import torch
import torch.nn.functional as F

def cdiff(x: torch.Tensor, dim: int, dx: float = 1.0) -> torch.Tensor:
    """Central difference with replicate padding. x: [..., H, W]. dim=-2 or -1."""
    if dim == -2:  # y/lat
        xpad = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return (xpad[..., 2:, :] - xpad[..., :-2, :]) / (2 * dx)
    if dim == -1:  # x/lon
        xpad = F.pad(x, (1, 1, 0, 0), mode="replicate")
        return (xpad[..., 2:] - xpad[..., :-2]) / (2 * dx)
    raise ValueError("dim must be -2 (y) or -1 (x)")

def laplacian(x: torch.Tensor, dy: float = 1.0, dx: float = 1.0) -> torch.Tensor:
    return cdiff(cdiff(x, -2, dy), -2, dy) + cdiff(cdiff(x, -1, dx), -1, dx)

def zscore_per_grid(x: torch.Tensor) -> torch.Tensor:
    """x: [T,H,W] -> per-grid z-scores over time axis (no temporal logic in decisions)."""
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd

def abs_lat_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """Return |lat| as [1,H,W], synthetic if you don’t have a lat field."""
    lat = torch.linspace(90.0, -90.0, H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
    return lat.abs().unsqueeze(0)

def spatial_percentile_batched(x: torch.Tensor, q: float) -> torch.Tensor:
    """x: [B,H,W] -> [B,1,1], percentile across space per frame."""
    B, H, W = x.shape
    return torch.quantile(x.view(B, H * W), q, dim=1, keepdim=True).unsqueeze(-1)