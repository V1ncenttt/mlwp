import numpy as np
from scipy.spatial import Voronoi
from skimage.draw import polygon
import torch
from models import FukamiNet, FukamiResNet, FukamiUNet, ReconstructionVAE
import numpy as np
from scipy.spatial import Voronoi
from skimage.draw import polygon

def voronoi_tesselate(sensor_coords, sensor_values, grid_shape):
    """
    Perform Voronoi tessellation of sensor values on a 2D grid.

    Args:
        sensor_coords: List of (row, col) tuples for sensor positions
        sensor_values: Array of values at those sensor locations (same order)
        grid_shape: Tuple (H, W) for grid size

    Returns:
        tessellated: 2D array of shape (H, W) with sensor values filled in
    """
    H, W = grid_shape
    # Convert (row, col) → (x, y) for Voronoi
    points = np.array([(x, y) for (y, x) in sensor_coords])
    vor = Voronoi(points)

    tessellated = np.zeros((H, W), dtype=np.float32)

    for i, region_idx in enumerate(vor.point_region):
        verts_idx = vor.regions[region_idx]
        if -1 in verts_idx or len(verts_idx) == 0:
            continue  # skip open cells
        verts = vor.vertices[verts_idx]
        rr, cc = polygon(verts[:, 1], verts[:, 0], shape=(H, W))  # y, x
        tessellated[rr, cc] = sensor_values[i]

    return tessellated

def sample_sensor_locations(grid_shape, num_sensors, seed=None):
    """
    Randomly sample sensor locations on a lat-lon grid.

    Args:
        grid_shape: Tuple (H, W) for the grid.
        num_sensors: Number of sensors to sample.
        seed: Optional random seed for reproducibility.

    Returns:
        List of (row, col) tuples for sensor locations.
    """
    H, W = grid_shape
    rng = np.random.default_rng(seed)

    # Flattened list of (row, col) indices
    indices = np.stack(np.meshgrid(np.arange(H), np.arange(W)), axis=-1).reshape(-1, 2)
    
    sampled = indices[rng.choice(indices.shape[0], size=num_sensors, replace=False)]
    return [tuple(coord) for coord in sampled]

def get_device():
    """
    Get the device to use for PyTorch.
    
    Returns:
        device: 'cuda' if available, otherwise 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_model(model):
    if model == "fukami":
        return FukamiNet()
    elif model == "fukami_resnet":
        return FukamiResNet()
    elif model == "fukami_unet":
        return FukamiUNet()
    elif model == "vae":
        # Assuming VAE is defined elsewhere
        return ReconstructionVAE(channels=2, latent_dim=128)
    else:
        raise ValueError(f"Unknown model type: {model}")
