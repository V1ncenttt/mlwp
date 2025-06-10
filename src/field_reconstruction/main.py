import os
import argparse
import yaml
import torch
import wandb
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import Voronoi
from skimage.draw import polygon

from utils import sample_sensor_locations, voronoi_tesselate
from models import FukamiNet, ReconstructionVAE
from train import train
from test import evaluate
from prepare_data import create_and_save_field_reco_dataset, load_field_reco_dataloaders

def get_config_params():
    """
    Load configuration parameters from a YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"📄 Loaded configuration for dataset: {config['data']['dataset']}")
    return config
def get_dataset_mode(model_name):
    """
    Get the dataset mode based on the model name.
    """
    if model_name in ["fukami", "vae", "fukami_resnet", "fukami_unet"]:
        return "voronoi"
    elif "vitae" in model_name or "cwgan" in model_name:
        return "vitae"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def main():
    print("🚀 Field Reconstruction Pipeline")

    parser = argparse.ArgumentParser(description="Field Reconstruction with Interpolation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--num_sensors", type=int, default=100, help="Number of sensors to sample")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model/save path")
    args = parser.parse_args()

    config = get_config_params()

    # Process variables
    if config["variables"] == "5vars_2d":
        variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "total_column_water_vapour"
        ]
    else:
        raise ValueError(f"Unknown variable set: {config['variables']}")

    percent = config["percent"]
    output_dir = config["output_dir"]
    batch_size = config["batch_size"]
    split_ratio = config["split_ratio"]
    model = config["model"]
    reco_mode = get_dataset_mode(model)
   

    dataset_train_path = os.path.join(
        output_dir,
        f"{'_'.join([v.replace('/', '_') for v in variables])}_{percent}p_{reco_mode}_{len(variables)}vars_train.pt"
    )
    
    # Dataset creation
    if not os.path.exists(dataset_train_path):
        print("🛠️ Dataset not found. Creating dataset...")
        create_and_save_field_reco_dataset(
            path_to_nc="../../data/weatherbench2_5vars_flat.nc",
            variables=variables,
            percent=percent,
            output_dir=output_dir,
            mode=reco_mode,
        )
    else:
        print("✅ Dataset already exists. Skipping dataset creation.")

    # Training or Testing
    if args.train:
        print(f"🚂 Training model: {model}...")

        train_loader, val_loader = load_field_reco_dataloaders(
            variable_list=variables,
            batch_size=batch_size,
            mode="train",
            percent=percent,
            split_ratio=split_ratio,
            reco_mode=reco_mode,
        )
        data = {"train_loader": train_loader, "val_loader": val_loader}
        train(model, data, "models/saves/", config)

    elif args.test:
        model_type = config["test"]["model_type"]
        model_path = config["test"]["model_path"]
        print(f"🧪 Testing model: {model_type}...")

        print(f"📊 Loading test data...")
        test_loader = load_field_reco_dataloaders(
            variable_list=variables,
            batch_size=batch_size,
            mode="test",
            percent=percent,
            reco_mode=reco_mode,
        )
        evaluate(model_type, test_loader, model_path, variables)

    else:
        print("⚠️ No action specified. Use --train or --test.")

if __name__ == "__main__":
    main()