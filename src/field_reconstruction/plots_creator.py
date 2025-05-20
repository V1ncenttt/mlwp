import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging

def kriging_interpolation(yx, values, H, W, model='exponential'):
    x = yx[:, 1].astype(np.float64)
    y = yx[:, 0].astype(np.float64)
    z = values.astype(np.float64)

    gridx = np.arange(W, dtype=float)
    gridy = np.arange(H, dtype=float)

    try:
        ok = OrdinaryKriging(x, y, z, variogram_model=model, verbose=False)
        interp, _ = ok.execute('grid', gridx, gridy)
    except Exception as e:
        print(f"Kriging failed: {e}")
        interp = np.zeros((H, W))  # fallback

    return interp

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
    
def plot_voronoi_reconstruction_comparison(voronoi_mask, ground_truth, cnn_output, mask=None,
                                            titles=("Voronoi Regions", "Ground Truth", "CNN Reconstruction"),
                                            save_path=None):
    """
    Plots Voronoi regions, ground truth, and CNN reconstruction side by side.

    Args:
        voronoi_mask: 2D array with integers representing Voronoi regions.
        ground_truth: 2D array of the real field values.
        cnn_output: 2D array of reconstructed field from CNN.
        mask: Optional 2D array to overlay sensor locations (1s).
        titles: Tuple of titles for each subplot.
        save_path: Optional path to save the figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Voronoi mask
    axs[0].imshow(voronoi_mask, cmap="tab20", interpolation="none")
    axs[0].set_title(titles[0])
    axs[0].axis("off")

    # Plot ground truth
    im1 = axs[1].imshow(ground_truth, cmap="coolwarm", interpolation="none")
    axs[1].set_title(titles[1])
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], shrink=0.75)

    # Plot CNN reconstruction
    im2 = axs[2].imshow(cnn_output, cmap="coolwarm", interpolation="none")
    axs[2].set_title(titles[2])
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2], shrink=0.75)

    # Overlay sensors if provided
    if mask is not None:
        y, x = np.where(mask > 0)
        axs[1].scatter(x, y, s=8, c='black', marker='x', label='Sensors')
        axs[2].scatter(x, y, s=8, c='black', marker='x')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"📸 Saved Voronoi + reconstruction plot to {save_path}")
    else:
        plt.show()

    plt.close()
    
    
def plot_l2_error_distributions(l2_errors_dict, variable_names, model_name, save_dir):
    """
    Create grid plot of L2 error distributions for each variable with mean line.

    Args:
        l2_errors_dict: dict where keys are variable names and values are arrays of L2 errors.
        variable_names: list of variable names to include in the plot (should match keys).
        model_name: string name of the model to include in the plot title.
        save_dir: path to save the resulting PNG file.
    """
    num_vars = len(variable_names)
    cols = 2
    rows = (num_vars + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axs = axs.flatten()
    
    for i, var in enumerate(variable_names):
        print(var)
        ax = axs[i]
        errors = np.array(l2_errors_dict[var])
        
        mean_err = np.mean(errors)
        median_err = np.median(errors)

        ax.hist(errors, bins=50, color='steelblue', edgecolor='black')
        ax.axvline(mean_err, color='red', linewidth=2)
        ax.text(mean_err, ax.get_ylim()[1] * 0.9, f"{mean_err*100:.2f}%", color='red', ha='center', fontsize=10)
        ax.text(median_err, ax.get_ylim()[1] * 0.8, f"{median_err*100:.2f}%", color='orange', ha='center', fontsize=10)
        ax.axvline(median_err, color='orange', linewidth=2)
        ax.set_title(var, fontsize=12)
        ax.set_xlabel("Relative L2 Error")
        ax.set_ylabel("Frequency")

    # Remove unused subplots if any
    for j in range(len(variable_names), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Relative L2 error — {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_l2_error_distributions.png")
    plt.savefig(save_path, dpi=200)
    print(f"📊 L2 error distribution plot saved to {save_path}")
    plt.close()
    


def plot_random_reconstruction(model, val_loader, device, model_name, save_dir, num_samples=7):
    """
    Plot reconstruction vs ground truth for multiple variables/channels.
    For each variable, generate a 7x3 grid: Ground Truth, Prediction, Error.
    Supports model-based and cubic interpolation.
    """
    model_is_interp = model_name == "cubic_interpolation"
    model_is_kriging = model_name == "kriging"
    

    if not model_is_interp and not model_is_kriging:
        model.eval()

    dataset = val_loader.dataset
    total_samples = len(dataset)
    input_sample, target_sample = dataset[0]
    num_channels = target_sample.shape[0]

    with torch.no_grad():
        for ch in range(num_channels):
            fig, axs = plt.subplots(num_samples, 3, figsize=(12, 2.2 * num_samples))
            fig.suptitle(f"Channel {ch}: Variable {ch}", fontsize=14)

            for i in range(num_samples):
                idx = random.randint(0, total_samples - 1)
                x, y = dataset[idx]
                gt = y[ch].cpu().numpy()

                if model_is_interp:
                    # x: (num_channels+1, H, W) -- [0]=mask, [1:]=Voronoi values
                    x_np = x.cpu().numpy()
                    mask = x_np[0]
                    tess = x_np[1 + ch]  # variable-specific Voronoi values
                    H, W = mask.shape
                    yx = np.argwhere(mask > 0)
                    values = tess[mask > 0]
                    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                    interp = griddata(yx, values, (grid_y, grid_x), method='cubic', fill_value=0.0)
                    pred = interp
                    
                elif model_is_kriging:
                    
                    x_np = x.cpu().numpy()
                    mask = x_np[0]
                    tess = x_np[1 + ch]  # variable-specific Voronoi values
                    H, W = mask.shape
                    yx = np.argwhere(mask > 0)
                    values = tess[mask > 0]
                    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                    interp = kriging_interpolation(yx, values, H, W, model='exponential')
                    pred = interp
                    
                else:
                    x_in = x.unsqueeze(0).to(device)
                    if model_name == "vae":
                        recon_x, mu, logvar = model(x_in)
                        pred = recon_x.squeeze().cpu().numpy()[ch]
                    else:
                        pred = model(x_in).squeeze().cpu().numpy()[ch]

                error = np.abs(gt - pred)

                axs[i, 0].imshow(gt, cmap='viridis')
                axs[i, 0].set_title(f"Image {i} — Ground Truth")
                axs[i, 0].axis("off")

                axs[i, 1].imshow(pred, cmap='viridis')
                axs[i, 1].set_title("Reconstruction" if not model_is_interp else "Cubic Interpolation")
                axs[i, 1].axis("off")

                axs[i, 2].imshow(error, cmap='hot')
                axs[i, 2].set_title("Abs Error")
                axs[i, 2].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            fname = os.path.join(save_dir, f"{model_name}_reco_channel_{ch}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"📸 Saved plot for channel {ch} to {fname}")
