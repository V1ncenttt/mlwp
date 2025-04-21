import os
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import yaml
import wandb
from scipy.interpolate import griddata
from scipy.spatial import Voronoi
from skimage.draw import polygon
import torch
from utils import sample_sensor_locations, voronoi_tesselate
from models import FukamiNet, ReconstructionVAE
from train import train
from test import evaluate
from prepare_data import create_and_save_field_reco_dataset, load_field_reco_dataloaders
from tqdm import tqdm

"""
# --- Load dataset ---
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/weatherbench2_5vars_flat.nc"))
print("🔍 Loading dataset...")
ds = xr.open_dataset(data_path)

# Pick a 2D variable and first timestep
var_name = "2m_temperature"
assert var_name in ds, f"{var_name} not found in dataset"
field = ds[var_name].isel(time=0).values  # shape: (lat, lon)

# Normalize field
field = (field - np.nanmean(field)) / np.nanstd(field)
H, W = field.shape
num_sensors = int(0.10 * H * W)

# --- Sample sensors ---
sensor_coords = sample_sensor_locations((H, W), num_sensors)
sensor_values = np.array([field[y, x] for (y, x) in sensor_coords])
points = np.array([(x, y) for (y, x) in sensor_coords])
grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

# --- Interpolation (linear and cubic) ---
print("📈 Performing interpolation...")
interp_linear = griddata(points, sensor_values, (grid_x, grid_y), method='linear', fill_value=np.nan)
interp_cubic = griddata(points, sensor_values, (grid_x, grid_y), method='cubic', fill_value=np.nan)

# --- Compute MSEs ---
true_tensor = torch.tensor(field, dtype=torch.float32)
linear_tensor = torch.tensor(interp_linear, dtype=torch.float32)
cubic_tensor = torch.tensor(interp_cubic, dtype=torch.float32)

mask_linear = ~torch.isnan(linear_tensor)
mask_cubic = ~torch.isnan(cubic_tensor)

mse_linear = torch.mean((true_tensor[mask_linear] - linear_tensor[mask_linear]) ** 2)
mse_cubic = torch.mean((true_tensor[mask_cubic] - cubic_tensor[mask_cubic]) ** 2)

print(f"✅ Linear MSE: {mse_linear.item():.5f}")
print(f"✅ Cubic  MSE: {mse_cubic.item():.5f}")

# --- Plotting ---
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../plots"))
os.makedirs(output_dir, exist_ok=True)

# Sparse sensor map
sparse_field = np.full_like(field, np.nan)
for (y, x), val in zip(sensor_coords, sensor_values):
    sparse_field[y, x] = val

# Voronoi tessellation
vor_mask = voronoi_mask(sensor_coords, (H, W))

fig, axs = plt.subplots(1, 5, figsize=(20, 4))

axs[0].imshow(field, cmap="coolwarm")
axs[0].set_title("Ground Truth")
axs[0].axis('off')

axs[1].imshow(sparse_field, cmap="coolwarm")
axs[1].set_title("Sparse Sensors (10%)")
axs[1].axis('off')

axs[2].imshow(interp_linear, cmap="coolwarm")
axs[2].set_title(f"Linear (MSE={mse_linear.item():.4f})")
axs[2].axis('off')

axs[3].imshow(interp_cubic, cmap="coolwarm")
axs[3].set_title(f"Cubic (MSE={mse_cubic.item():.4f})")
axs[3].axis('off')

axs[4].imshow(vor_mask, cmap="tab20")
axs[4].set_title("Voronoi Tessellation")
axs[4].axis('off')

plt.tight_layout()
filename = os.path.join(output_dir, f"interp_comparison_{var_name}.png")
plt.savefig(filename, dpi=200)
print(f"📸 Plot saved to {filename}")
plt.show()
"""
def get_train_objects(args):
    
    if args.model == "fukami":
        model = FukamiNet()
    elif args.model == "VAE":
        model = ReconstructionVAE(channels=2, latent_dim=128)
    
def get_config_params():
    # Load the configuration parameters from a YAML file
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    #Print the dataset name
    print(f"Dataset: {config['data']['dataset']}")
    return config
    
if __name__ == "__main__":
    print("🚀 Field Reconstruction.")
    # This block is executed when the script is run directly
    parser = argparse.ArgumentParser(description="Field Reconstruction with Interpolation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--num_sensors", type=int, default=100, help="Number of sensors to sample")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model/save path")
    
    args = parser.parse_args()
    config = get_config_params()
    
   
    variable = config["variable"]
    percent = config["percent"]
    output_dir = config["output_dir"]
    batch_size = config["batch_size"]
    split_ratio = config["split_ratio"]
    model = config["model"]
    
    #If the dataset is not already created, create it
    if not os.path.exists(os.path.join(output_dir, f"{variable}_{percent}p_train.pt")):
        print("Dataset not found. Creating dataset...")
        # Create the dataset
        create_and_save_field_reco_dataset(
        path_to_nc="../../data/weatherbench2_5vars_flat.nc",
        variable=variable,
        percent=10,
        output_dir="../../data/weatherbench2_fieldreco/"
        )
    else:
        print("Dataset already exists. Skipping dataset creation.")
        
    if args.train:
        # Call the training function
        print(f"Training model {model}...")
        
        train_dataloader, val_dataloader = load_field_reco_dataloaders(
            batch_size=batch_size,
            mode="train",
            variable=variable,
            percent=percent,
            split_ratio=split_ratio,
        )
        
        data = {"train_loader": train_dataloader, "val_loader": val_dataloader}
        train(model, data, "models/saves/", config)
        
    elif args.test:
        # Call the testing function
        model_type = config["test"]["model_type"]
        model_path = config["test"]["model_path"]
        print(f"Testing model {model_type}...")
        
        print(f"📊 Loading data")
        test_dataloader = load_field_reco_dataloaders(
            batch_size=batch_size,
            mode="test",
            variable=variable,
            percent=percent,
        )
        
        
        evaluate(model_type, test_dataloader, model_path)
    else:
        print("No action specified. Use --train or --test.")