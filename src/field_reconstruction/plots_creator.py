import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sparse_field(sensor_coords, sensor_values, grid_shape, title="Sparse Sensor Field", save_path=None):
    """
    Plots a sparse field (just the sensor values at sampled locations) on a colormap.

    Args:
        sensor_coords: List of (row, col) tuples.
        sensor_values: List or array of values associated with those sensors.
        grid_shape: Tuple (H, W) of the target grid.
        title: Plot title.
        save_path: Optional path to save the plot as PNG.
    """
    H, W = grid_shape
    sparse = np.full((H, W), np.nan)
    for (y, x), val in zip(sensor_coords, sensor_values):
        sparse[y, x] = val

    plt.figure(figsize=(6, 4))
    plt.imshow(sparse, cmap="coolwarm", interpolation="none")
    plt.title(title)
    plt.colorbar()
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"📸 Saved sparse field plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_reconstruction_comparison(ground_truth, reconstruction, mask=None, titles=("Ground Truth", "Reconstruction"), save_path=None):
    """
    Plots ground truth and reconstructed field side by side.

    Args:
        ground_truth: 2D numpy array.
        reconstruction: 2D numpy array.
        mask: Optional mask to overlay sensor locations.
        titles: Tuple of titles for each subplot.
        save_path: Optional path to save the plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for i, (field, title) in enumerate(zip([ground_truth, reconstruction], titles)):
        im = axs[i].imshow(field, cmap="coolwarm", interpolation="none")
        axs[i].set_title(title)
        axs[i].axis("off")
        fig.colorbar(im, ax=axs[i], shrink=0.75)

        # Overlay sensor locations if provided
        if mask is not None:
            y, x = np.where(mask > 0)
            axs[i].scatter(x, y, s=8, c='black', marker='x', label='Sensors')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"📸 Saved reconstruction plot to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()