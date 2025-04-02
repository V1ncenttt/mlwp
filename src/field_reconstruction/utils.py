import numpy as np
from scipy.spatial import Voronoi
from skimage.draw import polygon
import torch

def voronoi_mask(sensor_coords, grid_shape):
    """
    Create a Voronoi tessellation mask over a grid.
    
    Args:
        sensor_coords: array of shape (N, 2), sensor locations as (row, col)
        grid_shape: shape of the grid (H, W)

    Returns:
        voronoi_map: array of shape (H, W) with integers indicating Voronoi regions
    """
    points = np.array(sensor_coords)
    vor = Voronoi(points)

    H, W = grid_shape
    mask = np.zeros((H, W), dtype=np.int32)

    for i, region_idx in enumerate(vor.point_region):
        verts_idx = vor.regions[region_idx]
        if -1 in verts_idx or len(verts_idx) == 0:
            continue  # skip infinite regions
        verts = vor.vertices[verts_idx]
        rr, cc = polygon(verts[:, 1], verts[:, 0], shape=(H, W))
        mask[rr, cc] = i + 1

    return mask

def sample_sensor_locations(grid_shape, num_sensors):
    """
    Randomly sample sensor locations on a lat-lon grid.
    
    Returns a list of (row, col) tuples.
    """
    H, W = grid_shape
    indices = np.stack(np.meshgrid(np.arange(H), np.arange(W)), axis=-1).reshape(-1, 2)
    sampled = indices[np.random.choice(indices.shape[0], num_sensors, replace=False)]
    return [tuple(coord) for coord in sampled]

def get_device():
    """
    Get the device to use for PyTorch.
    
    Returns:
        device: 'cuda' if available, otherwise 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'