import xarray as xr
import wandb
import yaml
import os
import argparse

# Relative path to save location (your data/ folder)
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/weatherbench2_5vars_3d.nc"))

# Select 5 key surface variables
selected_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_column_water_vapour"
]

selected_vars_3D = ["u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "geopotential"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
    
    ds = xr.open_zarr(path, consolidated=True)
    
    print(ds)

    ds_subset = ds[selected_vars_3D]

    print(f"Saving subset to: {save_path}")
    ds_subset.to_netcdf(save_path)

    print("Done!")
