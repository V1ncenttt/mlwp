import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from models.diffusion.ddpm import DDIM, DDPM

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
        ax.legend(['Mean', 'Median'], loc='upper right')

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
    channel_names =["Temperature (10m)", "U component of Wind (10m)", "V component of Wind (10m)", "Water column vapour", "Sea level pressure"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_is_interp and not model_is_kriging:
        model.eval()

    dataset = val_loader.dataset
    total_samples = len(dataset)
    input_sample, target_sample = dataset[0]
    num_channels = target_sample.shape[0]

    with torch.no_grad():
        for ch in range(num_channels):
            fig, axs = plt.subplots(3, num_samples, figsize=(2.5 * num_samples, 6))
            fig.suptitle(f"{channel_names[ch]}", fontsize=14)

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
                    y_in = y.unsqueeze(0).to(device)
                    if model_name == "vae":
                        recon_x, mu, logvar = model(x_in)
                        pred = recon_x.squeeze().cpu().numpy()[ch]
                    elif "vitae" in model_name:
                        dex_x, enc_x = model(x_in)
                        pred = dex_x.squeeze().cpu().numpy()[ch]
                    elif "gan" in model_name:
                        #Remove first channel (mask) for GANs
                        x_in = x_in[:, 1:, :, :]
                        pred = model(x_in).squeeze().cpu().numpy()[ch]
                    elif "diffusion" in model_name:
                        if isinstance(model, DDIM):
                            pred = model.sample(n_sample=1,size=(y_in.shape[1], y_in.shape[2], y_in.shape[3]), device=device, cond=x_in, ddim_steps=20).squeeze().cpu().numpy()[ch]
                        elif isinstance(model, DDPM):
                            pred = model.sample(n_sample=1,size=(y_in.shape[1], y_in.shape[2], y_in.shape[3]), device=device, cond=x_in).squeeze().cpu().numpy()[ch]
                    else:
                        pred = model(x_in).squeeze().cpu().numpy()[ch]

                error = np.abs(gt - pred)

                im0 = axs[0, i].imshow(np.rot90(gt, k=-1), cmap='viridis')
                axs[0, i].set_title(f"Image {i}\nGround Truth", fontsize=8)
                axs[0, i].axis("off")
                fig.colorbar(im0, ax=axs[0, i], shrink=0.7, pad=0.03)

                im1 = axs[1, i].imshow(np.rot90(pred, k=-1), cmap='viridis')
                axs[1, i].set_title("Reconstruction", fontsize=8)
                axs[1, i].axis("off")
                fig.colorbar(im1, ax=axs[1, i], shrink=0.7, pad=0.03)

                im2 = axs[2, i].imshow(np.rot90(error, k=-1), cmap='hot')
                axs[2, i].set_title("Abs Error", fontsize=8)
                axs[2, i].axis("off")
                fig.colorbar(im2, ax=axs[2, i], shrink=0.7, pad=0.03)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            fname = os.path.join(save_dir, f"{model_name}_reco_channel_{ch}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"📸 Saved plot for channel {ch} to {fname}")
def plot_reconstruction_per_snapshot_per_field(
    model, val_loader, device, model_name, save_dir, indices,
    transpose_hw: bool = False,         # set True only if arrays come as (W,H) and you need (H,W)
    rotate_k: int = 1,                  # 1=90° CCW, 3=90° CW. Pick the one that gives 64x32
    cmap: str = "viridis",
    gan_k: int = 8,                     # <-- # stochastic samples for GAN, averaged
    gan_seed: int | None = None,        # set for reproducibility
    save_gt: bool = True,               # <-- also save ground truth frames
    gt_prefix: str = "gt",              # <-- filename prefix for GT
):
    """
    For each (idx, channel), save TWO PNGs:
      (1) prediction only  -> {save_dir}/{model_name}__idx{idx}__ch{ch}.png
      (2) ground truth     -> {save_dir}/{gt_prefix}__idx{idx}__ch{ch}.png   (if save_gt)
    - No titles, no colorbar, no axes, no whitespace bars.
    - Orientation: optional transpose + rotate_k (use rotate_k=1 or 3 to swap 32x64 <-> 64x32).
    - Shared per-channel color scale computed from GT over the chosen indices.
    - GAN path: removes mask channel if present, adds a noise channel, does 'gan_k' stochastic
      forward passes and averages them.
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    model_is_interp = model_name == "cubic_interpolation"
    model_is_kriging = model_name == "kriging"
    model_is_gan = "gan" in model_name.lower()
    model_is_diff = "diffusion" in model_name.lower()
    model_is_cnn = not (model_is_gan or model_is_diff or model_is_interp or model_is_kriging)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not (model_is_interp or model_is_kriging):
        model.eval()

    if gan_seed is not None:
        torch.manual_seed(gan_seed)

    dataset = val_loader.dataset
    # infer num channels from first sample's target
    s0 = dataset[0]
    y0 = s0["y"] if isinstance(s0, dict) else s0[1]
    num_channels = y0.shape[0]

    @torch.no_grad()
    def _predict_full(x, y) -> np.ndarray:
        """Return prediction for all channels as [C,H,W] np.float32."""
        if model_is_interp or model_is_kriging:
            x_np = x.cpu().numpy()
            mask = x_np[0]
            C    = y.shape[0]
            H, W = mask.shape
            yx   = np.argwhere(mask > 0)
            gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            pred = np.zeros((C, H, W), dtype=np.float32)
            for ch in range(C):
                vals = x_np[1 + ch][mask > 0]
                if model_is_interp:
                    pred[ch] = griddata(yx, vals, (gy, gx), method="cubic", fill_value=0.0).astype(np.float32)
                else:
                    # Use your existing kriging_interpolation implementation
                    pred[ch] = kriging_interpolation(yx, vals, H, W, model="exponential").astype(np.float32)
            return pred

        # torch model paths
        x_in = x.unsqueeze(0).to(device)  # [1,Cx,H,W] (Cx could include mask)
        y_in = y.unsqueeze(0).to(device)  # [1,Cy,H,W]

        # ---- GAN: remove mask ch if present, add noise ch, average over gan_k samples ----
        if model_is_gan:
            x_mod = x_in
            if x_mod.shape[1] == y_in.shape[1] + 1:
                x_mod = x_mod[:, 1:, :, :]  # drop mask channel

            preds_accum = 0.0
            for _ in range(gan_k):
                z = torch.randn_like(x_mod[:, 0:1, :, :])  # (B,1,H,W)
                x_g = torch.cat([x_mod, z], dim=1)         # (B,Cy+1,H,W)
                yhat = model(x_g)
                if isinstance(yhat, (tuple, list)):
                    yhat = yhat[0]
                preds_accum += yhat
            yhat = preds_accum / float(gan_k)

        else:
            x_for = x_in
            # some GAN-style models may still need mask removal—kept for safety
            if x_for.shape[1] == y_in.shape[1] + 1 and not model_is_diff and not model_is_cnn:
                x_for = x_for[:, 1:, :, :]

            if model_is_diff:
                # Assume DDIM/DDPM symbols exist in your environment just like before
                if isinstance(model, DDIM):
                    yhat = model.sample(
                        n_sample=1,
                        size=(y_in.shape[1], y_in.shape[2], y_in.shape[3]),
                        device=device, cond=x_in, ddim_steps=20
                    )
                elif isinstance(model, DDPM):
                    yhat = model.sample(
                        n_sample=1,
                        size=(y_in.shape[1], y_in.shape[2], y_in.shape[3]),
                        device=device, cond=x_in
                    )
                else:
                    yhat = model(x_for)
            else:
                # try common signatures
                tried = [
                    lambda: model(x_for),
                    lambda: model(x_in, None),
                    lambda: model(x_in, mask=None),
                ]
                yhat = None
                for fn in tried:
                    try:
                        yhat = fn()
                        break
                    except Exception:
                        continue
                if yhat is None:
                    yhat = model(x_for)

        if isinstance(yhat, (tuple, list)):
            yhat = yhat[0]
        return yhat.squeeze(0).detach().cpu().float().numpy()

  
        # ---------- shared per-channel vmin/vmax from GT on the chosen indices ----------
    vlims = {}
    with torch.no_grad():
        for ch in range(num_channels):
            arrs = []
            for idx in indices:
                s = dataset[idx]
                y = s["y"] if isinstance(s, dict) else s[1]
                a = y[ch].cpu().numpy()
                if transpose_hw: 
                    a = a.T
                if rotate_k:     
                    a = np.rot90(a, k=rotate_k)
                arrs.append(a)

            stack = np.stack(arrs, axis=0)
            vmin, vmax = float(np.nanmin(stack)), float(np.nanmax(stack))
            vlims[ch] = (vmin, vmax)

            # 🔹 print per-channel limits for later colorbars
            print(f"Channel {ch}: vmin={vmin:.5f}, vmax={vmax:.5f}")
    os.makedirs(save_dir, exist_ok=True)

    # ---------- save PNGs per (idx, ch) ----------
    with torch.no_grad():
        for idx in indices:
            s = dataset[idx]
            x, y = (s["x"], s["y"]) if isinstance(s, dict) else (s[0], s[1])
            pred = _predict_full(x, y)  # [C,H,W]

            for ch in range(num_channels):
                vmin, vmax = vlims[ch]

                # --- prediction ---
                img_pred = pred[ch]
                if transpose_hw: img_pred = img_pred.T
                if rotate_k:     img_pred = np.rot90(img_pred, k=rotate_k)

                Hp, Wp = img_pred.shape
                fig = plt.figure(figsize=(Wp/25, Hp/25))
                ax = plt.axes([0, 0, 1, 1])
                ax.imshow(img_pred, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
                ax.axis("off")
                out_pred = os.path.join(save_dir, f"{model_name}__idx{idx}__ch{ch}.png")
                fig.savefig(out_pred, dpi=180, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # --- ground truth (optional) ---
                if save_gt:
                    img_gt = y[ch].cpu().numpy()
                    if transpose_hw: img_gt = img_gt.T
                    if rotate_k:     img_gt = np.rot90(img_gt, k=rotate_k)

                    Hg, Wg = img_gt.shape
                    fig2 = plt.figure(figsize=(Wg/25, Hg/25))
                    ax2 = plt.axes([0, 0, 1, 1])
                    ax2.imshow(img_gt, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
                    ax2.axis("off")
                    out_gt = os.path.join(save_dir, f"{gt_prefix}__idx{idx}__ch{ch}.png")
                    fig2.savefig(out_gt, dpi=180, bbox_inches="tight", pad_inches=0)
                    plt.close(fig2)