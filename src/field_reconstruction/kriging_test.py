import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from torch.utils.data import TensorDataset, DataLoader

output_dir = "../../data/weatherbench2_fieldreco/"
base_name = "2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind_mean_sea_level_pressure_total_column_water_vapour_10p_voronoi_5vars"
batch_size = 32
path = os.path.join(output_dir, f"{base_name}_test.pt")
data = torch.load(path)
dataset = TensorDataset(data["X"], data["Y"])
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Pick a random sample
idx = np.random.randint(0, len(dataset))
x, y = dataset[idx]  # x: (C, H, W), y: (C, H, W)

print(len(x), len(y))
feature_idx = 0  # Change this to select a different physical variable
mask = x[0].numpy()
values = x[1 + feature_idx].numpy()  # voronoi value for selected variable
field = y[feature_idx].numpy()

H, W = mask.shape
coords = np.argwhere(mask > 0)
sensor_values = values[mask > 0]

# Extract x, y positions
xs = coords[:, 1].astype(np.float64)
ys = coords[:, 0].astype(np.float64)
zs = sensor_values.astype(np.float64)
print(len(xs), len(ys), len(zs))


gridx = np.arange(W, dtype='float64')
gridy = np.arange(H, dtype='float64')

print("Performing Ordinary Kriging...")
try:
    ok = OrdinaryKriging(xs, ys, zs, variogram_model='exponential', verbose=False, enable_plotting=True)
    print(ok.variogram_model_parameters)
    z_interp, _ = ok.execute("grid", gridx, gridy)
except Exception as e:
    print(f"Kriging failed: {e}")
    z_interp = np.zeros((H, W))

# Plotting
sparse_field = np.full((H, W), np.nan)
for (y_coord, x_coord), val in zip(coords, sensor_values):
    sparse_field[y_coord, x_coord] = val

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Compute global vmin and vmax for consistent color scale
all_data = [field, sparse_field, z_interp]
vmin = min(np.nanmin(d) for d in all_data)
vmax = max(np.nanmax(d) for d in all_data)

im0 = axs[0].imshow(field, cmap="coolwarm", vmin=vmin, vmax=vmax)
axs[0].set_title("Ground Truth")
axs[0].axis("off")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(sparse_field, cmap="coolwarm", vmin=vmin, vmax=vmax)
axs[1].set_title("Sparse Inputs")
axs[1].axis("off")
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(z_interp, cmap="coolwarm", vmin=vmin, vmax=vmax)
axs[2].set_title("Kriging Interpolation")
axs[2].axis("off")
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = os.path.join("plots", f"kriging_result_feature{feature_idx}.png")
os.makedirs("plots", exist_ok=True)
plt.savefig(save_path, dpi=200)
print(f"📸 Saved kriging interpolation plot to {save_path}")
plt.close()