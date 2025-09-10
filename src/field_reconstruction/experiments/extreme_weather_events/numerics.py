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


import math
import torch

def global_quantile_hist(
    x: torch.Tensor,
    q: float,
    *,
    transform: str | None = None,   # None | "abs"
    bins: int = 4096,
    chunk_elems: int = 2_000_000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Robust, streaming, *approximate* global quantile over all elements of x.
    - Two-pass histogram: (1) get global min/max, (2) accumulate histogram over chunks.
    - Then interpolate within the quantile bin.
    - Works for big tensors; low memory; avoids full sort.

    Args:
      x: arbitrary-shape tensor (will be flattened lazily)
      q: quantile in [0,1]
      transform: if "abs", uses |x| values
      bins: histogram bins (increase for tighter thresholds, small extra cost)
      chunk_elems: process this many scalars per chunk
      device/dtype: staging for hist; keep on CPU to free GPU

    Returns:
      float threshold (Python scalar)
    """
    assert 0.0 <= q <= 1.0
    # Lazy flattened iterator over chunks
    flat = x.reshape(-1)

    # 1) global min/max (streaming)
    n = flat.numel()
    if n == 0:
        return float("nan")

    # Use a small sample to guard degenerate ranges
    sample_n = min(n, 1_000_000)
    sample = flat[:sample_n]
    if transform == "abs":
        sample = sample.abs()
    s_min = sample.min().item()
    s_max = sample.max().item()

    # Expand min/max over all chunks
    gmin, gmax = s_min, s_max
    for start in range(sample_n, n, chunk_elems):
        end = min(start + chunk_elems, n)
        chunk = flat[start:end]
        if transform == "abs":
            chunk = chunk.abs()
        cmin = chunk.min().item()
        cmax = chunk.max().item()
        if cmin < gmin: gmin = cmin
        if cmax > gmax: gmax = cmax

    if not math.isfinite(gmin) or not math.isfinite(gmax) or gmin == gmax:
        # Degenerate: all equal (or NaNs already filtered out elsewhere)
        return float(gmin)

    # 2) accumulate histogram over chunks (CPU)
    edges = torch.linspace(gmin, gmax, bins + 1, dtype=dtype, device=device)
    counts = torch.zeros(bins, dtype=torch.float64, device=device)
    total = 0

    # helper to histogram a chunk using searchsorted (fast, portable)
    def _accumulate_hist(chunk_):
        nonlocal counts, total
        if transform == "abs":
            chunk_ = chunk_.abs()
        # bucket indices in [0, bins-1]; clamp right edge
        idx = torch.searchsorted(edges, chunk_.to(dtype), right=False) - 1
        idx = idx.clamp_(0, bins - 1)
        # bincount on CPU
        counts.index_add_(0, idx.to(counts.device, non_blocking=True), torch.ones_like(idx, dtype=counts.dtype, device=counts.device))
        total += chunk_.numel()

    # first the sample we already have
    _accumulate_hist(sample)

    for start in range(sample_n, n, chunk_elems):
        end = min(start + chunk_elems, n)
        _accumulate_hist(flat[start:end])

    if total == 0:
        return float("nan")

    # 3) find q-bin and interpolate
    cumsum = torch.cumsum(counts, dim=0)
    target = q * total
    bin_idx = int(torch.searchsorted(cumsum, torch.tensor(target, dtype=cumsum.dtype, device=cumsum.device)).item())
    bin_idx = max(0, min(bin_idx, bins - 1))

    prev_cum = cumsum[bin_idx - 1].item() if bin_idx > 0 else 0.0
    in_bin = counts[bin_idx].item()
    if in_bin <= 0:
        # empty bin (rare with high bins); fall back to bin edge
        return float(edges[bin_idx + 1].item())

    frac = (target - prev_cum) / in_bin
    left = edges[bin_idx].item()
    right = edges[bin_idx + 1].item()
    return float(left + frac * (right - left))