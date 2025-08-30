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
from experiments.sparsity.sparsity_datasets_generator import generate_sparsity_datasets
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
    if model_name in ["fukami", "vae", "fukami_resnet", "fukami_unet", "vt_unet"] or "diffusion" in model_name.lower():
        return "voronoi"
    elif "vitae" in model_name or "cwgan" in model_name:
        return "vitae"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
  
def check_sparsity_datasets_exist():
    """
    Check if sparsity datasets exist in the expected directory.
    Returns True if all expected datasets are found, False otherwise.
    """
    datasets_loc = "../../data/weatherbench2_fieldreco/sparsity/"
    
    # Expected percentages and modes
    percentages = [0.5, 1, 2, 5, 7.5, 20, 30, 50]
    modes = ["voronoi", "vitae"]
    
    # Variable prefix for filename
    variables_prefix = "2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind_mean_sea_level_pressure_total_column_water_vapour"
    
    # Check if directory exists
    if not os.path.exists(datasets_loc):
        print(f"❌ Sparsity datasets directory does not exist: {datasets_loc}")
        return False
    
    missing_files = []
    total_expected = len(percentages) * len(modes)  # 6 percentages × 2 modes = 12 files
    found_files = 0
    
    print(f"🔍 Checking for sparsity datasets in {datasets_loc}")
    
    for percentage in percentages:
        for mode in modes:
            # Construct expected filename
            filename = f"{variables_prefix}_{percentage}p_{mode}_5vars_test.pt"
            filepath = os.path.join(datasets_loc, filename)
            
            if os.path.exists(filepath):
                found_files += 1
                print(f"✅ Found: {filename}")
            else:
                missing_files.append(filename)
                print(f"❌ Missing: {filename}")
    
    print(f"\n📊 Summary: Found {found_files}/{total_expected} sparsity datasets")
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All sparsity datasets found!")
        return True
def main():
    print("🚀 Field Reconstruction Pipeline")

    parser = argparse.ArgumentParser(description="Field Reconstruction with Interpolation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--num_sensors", type=int, default=100, help="Number of sensors to sample")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model/save path")
    parser.add_argument("--sparsity", action="store_true", help="Generate sparsity datasets")
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
        evaluate(model_type, test_loader, model_path, variables, config)
    elif args.sparsity:
        print(f"📊 Generating sparsity datasets...")
        if not check_sparsity_datasets_exist():
            print("❌ Some sparsity datasets are missing. Generating them...")
            generate_sparsity_datasets()
        else:
            print("✅ All sparsity datasets already exist. Skipping generation.")
            # If you want to regenerate, uncomment the next line
        
        rrmses = []
        
        for sparsity in [0.5, 1, 2, 5, 7.5, 20, 30, 50]:
            model_type = config["test"]["model_type"]
            model_path = config["test"]["model_path"]
            print(f"🧪 Testing model: {model_type}...")

            print(f"📊 Loading test data...")
            test_loader = load_field_reco_dataloaders(
                variable_list=variables,
                batch_size=batch_size,
                mode="test",
                percent=sparsity,
                reco_mode=reco_mode,
                data_dir="../../data/weatherbench2_fieldreco/sparsity/"
            )
            
            perf = evaluate(model_type, test_loader, model_path, variables, config)
            rrmses.append(perf["rrmse"])
            
            print(f"Sparsity: {sparsity}%, RRMSE: {perf['rrmse']:.4f}")
        
        print("\n📊 Sparsity vs RRMSE:")
        for sparsity, rrmse in zip([0.5, 1, 2, 5, 7.5, 20], rrmses):
            print(f"Sparsity: {sparsity}%, RRMSE: {rrmse:.4f}")

    else:
        print("⚠️ No action specified. Use --train or --test.")

if __name__ == "__main__":
    main()