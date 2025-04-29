import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from scipy.spatial import Voronoi
from skimage.draw import polygon
from tqdm import tqdm
from utils import sample_sensor_locations, voronoi_tesselate

def create_and_save_field_reco_dataset(
    path_to_nc,
    variables=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_column_water_vapour"
    ],
    mode="voronoi",
    percent=10,
    train_split=0.8,
    output_dir="../../data/weatherbench2_fieldreco/",
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    print("🔍 Loading data...")
    ds = xr.open_dataset(path_to_nc)
    
    T, H, W = ds[variables[0]].shape
    n_sensors = int((percent / 100) * H * W)
    print(f"Using {n_sensors} sensors per timestep ({percent}%)")
    
    X = []
    Y = []

    rng = np.random.default_rng(seed)
    print("🔄 Normalizing and preparing samples...")
    for t in tqdm(range(T - 1), desc="Preparing samples"):
        fields = []
        for var in variables:
            field = ds[var].isel(time=t).values
            field = (field - np.nanmean(field)) / np.nanstd(field)
            field = np.nan_to_num(field, nan=0.0)
            fields.append(field)
        
        sensor_coords = sample_sensor_locations((H, W), n_sensors, seed=seed + t)
        
        print(f"🔄 Using mode {mode}")
        if mode == "voronoi":
            
            tess_fields = []
            for field in fields:
                sensor_values = np.array([field[y, x] for (y, x) in sensor_coords])
                tess = voronoi_tesselate(sensor_coords, sensor_values, (H, W))
                tess_fields.append(tess)
            
            mask = np.zeros((H, W), dtype=np.float32)
            for y, x in sensor_coords:
                mask[y, x] = 1.0

            x = np.stack([mask] + tess_fields, axis=0)  # (1+Nvars, H, W)
            y = np.stack(fields, axis=0)                # (Nvars, H, W)
        
        elif mode == "vitae":
            print
            I_fields = []
            for field in fields:
                I = np.zeros((H, W), dtype=np.float32)
                for (y, x) in sensor_coords:
                    I[y, x] = field[y, x]
                I_fields.append(I)

            mask = np.zeros((H, W), dtype=np.float32)
            for y, x in sensor_coords:
                mask[y, x] = 1.0

            x = np.stack(I_fields + [mask], axis=0)  # (Nvars+1, H, W)
            y = np.stack(fields, axis=0)             # (Nvars, H, W)
        
        else:
            raise ValueError("Unknown mode: choose 'voronoi' or 'vitae'.")

        X.append(torch.tensor(x, dtype=torch.float32))
        Y.append(torch.tensor(y, dtype=torch.float32))

    X = torch.stack(X)
    Y = torch.stack(Y)

    print("🔀 Splitting into train/test...")
    total = len(X)
    n_train = int(train_split * total)
    n_test = total - n_train

    train_X, test_X = torch.split(X, [n_train, n_test])
    train_Y, test_Y = torch.split(Y, [n_train, n_test])

    train_data = {"X": train_X, "Y": train_Y}
    test_data = {"X": test_X, "Y": test_Y}

    var_tag = "_".join([v.replace("/", "_") for v in variables])
    base_name = f"{var_tag}_{percent}p_{mode}_{len(variables)}vars"

    train_path = os.path.join(output_dir, f"{base_name}_train.pt")
    test_path = os.path.join(output_dir, f"{base_name}_test.pt")

    print(f"💾 Saving to {train_path} and {test_path}...")
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    print("✅ Done.")

def load_field_reco_dataloaders(
    variable_list,
    mode="train",
    reco_mode="voronoi",
    percent=10,
    batch_size=32,
    split_ratio=0.9,
    seed=42,
    data_dir="../../data/weatherbench2_fieldreco/"
):
    var_tag = "_".join([v.replace("/", "_") for v in variable_list])
    base_name = f"{var_tag}_{percent}p_{reco_mode}_{len(variable_list)}vars"

    if mode == "train":
        path = os.path.join(data_dir, f"{base_name}_train.pt")
        data = torch.load(path)
        dataset = TensorDataset(data["X"], data["Y"])

        total = len(dataset)
        n_train = int(total * split_ratio)
        n_val = total - n_train
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    elif mode == "test":
        path = os.path.join(data_dir, f"{base_name}_test.pt")
        data = torch.load(path)
        dataset = TensorDataset(data["X"], data["Y"])
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    else:
        raise ValueError("mode must be 'train' or 'test'")