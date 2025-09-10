"""
This script is used to generate datasets for sparsity experiments in field reconstruction.
From the  test voronoi/vitae datasets with 10% data, it generates datasets with varying sparsity levels.
The generated datasets are saved in the specified output directory.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directories to path to import utils
sys.path.append('../../')
from utils import sample_sensor_locations, voronoi_tesselate


def generate_specific_sparsity_datasets(input_file, output_dir, percentages, mode):
    """
    Generates datasets with varying sparsity levels from the input file.
    
    Args:
        input_file (str): Path to the input dataset file.
        output_dir (str): Directory where the generated datasets will be saved.
        percentages (list): List of sparsity percentages to generate datasets for.
        mode (str): Either "voronoi" or "vitae"
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🔍 Loading data from {input_file}...")
    data = torch.load(input_file)
    dataset = TensorDataset(data["X"], data["Y"])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Process one sample at a time
    
    # Extract filename components for naming
    base_filename = os.path.basename(input_file)
    filename_parts = base_filename.replace('.pt', '').split('_')
    variables_part = '_'.join(filename_parts[:-4])  # Remove _10p_mode_5vars_test
    
    for percentage in percentages:
        print(f"🔄 Generating {percentage}% sparsity dataset in {mode} mode...")
        
        X_new = []
        Y_new = []
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc=f"Processing {percentage}%")):
            # Get original dimensions
            B, C_in, H, W = inputs.shape
            _, C_out, _, _ = targets.shape
            
            # Calculate number of sensors for this sparsity level
            n_sensors = int((percentage / 100) * H * W)
            
            # Sample new sensor locations
            seed = 42 + batch_idx + int(percentage * 1000)  # Ensure reproducible but different seeds
            sensor_coords = sample_sensor_locations((H, W), n_sensors, seed=seed)
            
            if mode == "voronoi":
                # Extract the 5 variables from targets
                fields = [targets[0, v, :, :].numpy() for v in range(C_out)]
                
                # Create Voronoi tessellation for each variable
                tess_fields = []
                for field in fields:
                    sensor_values = np.array([field[y, x] for (y, x) in sensor_coords])
                    tess = voronoi_tesselate(sensor_coords, sensor_values, (H, W))
                    tess_fields.append(tess)
                
                # Create mask indicating sensor locations
                mask = np.zeros((H, W), dtype=np.float32)
                for y, x in sensor_coords:
                    mask[y, x] = 1.0
                
                # Stack: [mask, tess_field1, tess_field2, ..., tess_field5]
                x_new = np.stack([mask] + tess_fields, axis=0)  # (1+Nvars, H, W)
                
            elif mode == "vitae":
                # Extract the 5 variables from targets
                fields = [targets[0, v, :, :].numpy() for v in range(C_out)]
                
                # Create sparse fields (values only at sensor locations, 0 elsewhere)
                I_fields = []
                for field in fields:
                    I = np.zeros((H, W), dtype=np.float32)
                    for (y, x) in sensor_coords:
                        I[y, x] = field[y, x]
                    I_fields.append(I)
                
                # Create mask indicating sensor locations
                mask = np.zeros((H, W), dtype=np.float32)
                for y, x in sensor_coords:
                    mask[y, x] = 1.0
                
                # Stack: [sparse_field1, sparse_field2, ..., sparse_field5, mask]
                x_new = np.stack(I_fields + [mask], axis=0)  # (Nvars+1, H, W)
            
            else:
                raise ValueError("Mode must be either 'voronoi' or 'vitae'")
            
            # Convert to tensor and add to lists
            X_new.append(torch.tensor(x_new, dtype=torch.float32))
            Y_new.append(targets[0])  # Keep original targets
        
        # Stack all samples
        X_new = torch.stack(X_new)
        Y_new = torch.stack(Y_new)
        
        # Create output filename
        output_filename = f"{variables_part}_{percentage}p_{mode}_5vars_test.pt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save dataset
        new_data = {"X": X_new, "Y": Y_new}
        torch.save(new_data, output_path)
        print(f"💾 Saved {len(X_new)} samples to {output_path}")

        
def generate_sparsity_datasets():
    """Main function to generate all sparsity datasets"""
    # Base directory
    base_dir = "../../data/weatherbench2_fieldreco/"
    output_dir = "../../data/weatherbench2_fieldreco/sparsity/"
    
    # Input files
    test_10p_voronoi_file = os.path.join(base_dir, "2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind_mean_sea_level_pressure_total_column_water_vapour_10p_voronoi_5vars_test.pt")
    test_10p_vitae_file = os.path.join(base_dir, "2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind_mean_sea_level_pressure_total_column_water_vapour_10p_vitae_5vars_test.pt")
    
    # Sparsity percentages to generate
    percentages = [0.5, 1, 2, 5, 7.5, 20, 30, 50]
    
    print("🚀 Starting sparsity dataset generation...")
    
    # Generate Voronoi datasets
    print("\n📊 Generating Voronoi datasets...")
    if os.path.exists(test_10p_voronoi_file):
        generate_specific_sparsity_datasets(test_10p_voronoi_file, output_dir, percentages, "voronoi")
    else:
        print(f"❌ Voronoi input file not found: {test_10p_voronoi_file}")
    
    # Generate ViTAE datasets
    print("\n📊 Generating ViTAE datasets...")
    if os.path.exists(test_10p_vitae_file):
        generate_specific_sparsity_datasets(test_10p_vitae_file, output_dir, percentages, "vitae")
    else:
        print(f"❌ ViTAE input file not found: {test_10p_vitae_file}")
    
    print("\n✅ All sparsity datasets generated successfully!")

