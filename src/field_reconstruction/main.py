import xarray as xr
import wandb
import yaml
import os

# Relative path to save location (your data/ folder)
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/weatherbench2_5vars_2019_2020.nc"))

# Select 5 key surface variables
selected_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_column_water_vapour"
]

if __name__ == "__main__":
    path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
    
    ds = xr.open_zarr(path, consolidated=True)
    
    print(ds)

    ds_subset = ds[selected_vars].sel(
    time=slice("2019-01-01", "2020-12-31")  # adjust this if needed
    )

    print(f"Saving subset to: {save_path}")
    ds_subset.to_netcdf(save_path)

    print("Done!")
