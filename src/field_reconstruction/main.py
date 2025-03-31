import xarray as xr
import wandb
import yaml

if __name__ == "__main__":
    path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'

    ds = xr.open_zarr(path, consolidated=True)

    print(ds)
    print("Done!")
