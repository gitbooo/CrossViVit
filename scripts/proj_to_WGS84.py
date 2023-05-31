"""Satellite loader"""
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import time
import warnings
from pathlib import Path

import dask
import hydra
import numpy as np
import pandas as pd
import pyresample
import xarray as xr
from ocf_blosc2 import Blosc2
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")
dask.config.set({"array.slicing.split_large_chunks": False})


def setup_dataset(dataset):
    """
    Setup the dataset and make sure everything is working.
    Code source: https://github.com/openclimatefix/ocf_datapipes/blob/8fed3089589fb7db25cdcfba53418db638201b5a/ocf_datapipes/load/satellite.py#L16
    """
    if "x_geostationary_coordinates" in dataset:
        del dataset["x_geostationary_coordinates"]
        del dataset["y_geostationary_coordinates"]
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channel"})
    if "channels" in dataset:
        dataset = dataset.rename({"channels": "channel"})
    elif "channel" not in dataset:
        # This is HRV version 3, which doesn't have a channels dim.  So add one.
        dataset = dataset.expand_dims(dim={"channel": ["HRV"]}, axis=1)
    dataset = dataset.rename(
        {
            "time": "time_utc",
        }
    )
    if "y" in dataset.coords.keys():
        dataset = dataset.rename(
            {
                "y": "y_geostationary",
            }
        )

    if "x" in dataset.coords.keys():
        dataset = dataset.rename(
            {
                "x": "x_geostationary",
            }
        )

    if dataset.y_geostationary[0] < dataset.y_geostationary[-1]:
        dataset = dataset.reindex(y_geostationary=dataset.y_geostationary[::-1])
    if dataset.x_geostationary[0] > dataset.x_geostationary[-1]:
        dataset = dataset.reindex(x_geostationary=dataset.x_geostationary[::-1])

    data_array = dataset["data"]
    del dataset

    assert data_array.y_geostationary[0] > data_array.y_geostationary[-1]
    assert data_array.x_geostationary[0] < data_array.x_geostationary[-1]
    if "y_osgb" in data_array.dims:
        assert data_array.y_osgb[0, 0] > data_array.y_osgb[-1, 0]
        assert data_array.x_osgb[0, 0] < data_array.x_osgb[0, -1]

    data_array = data_array.transpose(
        "time_utc", "channel", "y_geostationary", "x_geostationary"
    )
    assert data_array.dims == (
        "time_utc",
        "channel",
        "y_geostationary",
        "x_geostationary",
    )
    datetime_index = pd.DatetimeIndex(data_array.time_utc)
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing
    # Satellite datetimes can sometimes be 04, 09, minutes past the hour, or other slight offsets.
    # These slight offsets will break downstream code, which expects satellite data to be at
    # exactly 5 minutes past the hour.
    assert (datetime_index == datetime_index.round("5T")).all()

    return data_array


def reproject(darray, geos_area_def, area_def):

    T, C, _, _ = darray.shape

    x = []
    for k in range(T):
        z = darray[k]
        xx = []
        for j in range(C):
            topo_image = pyresample.image.ImageContainerQuick(z[j], geos_area_def)
            topo_image_resampled = topo_image.resample(area_def)
            xx.append(topo_image_resampled.image_data)
        xx = np.stack(xx, axis=0)
        x.append(xx)

    x = np.stack(x, axis=0)

    return x


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="proj_to_WGS84.yaml"
)
def main(cfg: DictConfig) -> None:
    # Setup dataset
    zarr_path = Path(cfg.path_zarr)
    dataset = (
        xr.open_dataset(zarr_path, engine="zarr", chunks={})
        .drop_duplicates("time")
        .sortby("time")
    )
    data_array = setup_dataset(dataset)

    # Keep only every 30 minutes
    datetime_index = pd.DatetimeIndex(data_array.time_utc)
    start_time = datetime_index[0].floor("H")
    end_time = datetime_index[-1].ceil("H")
    time_range = pd.date_range(start=start_time, end=end_time, freq="30min")
    data_array = data_array.where(data_array.time_utc.isin(time_range), drop=True)

    # Setup projection parameters
    height, width = cfg.resolution
    lat = np.linspace(
        start=cfg.upper_right_coords[1], stop=cfg.lower_left_coords[1], num=height
    )
    lon = np.linspace(
        start=cfg.lower_left_coords[0], stop=cfg.upper_right_coords[0], num=width
    )
    _, _, H, W = data_array.shape
    geo_proj_params = OmegaConf.to_container(
        cfg.geostationary_proj_params, resolve=True
    )
    geos_area_def = pyresample.create_area_def(
        area_id="geos",
        projection=geo_proj_params,
        shape=(H, W),  # y, x
        area_extent=(
            data_array.x_geostationary[0].item(),  # lower_left_x
            data_array.y_geostationary[-1].item(),  # lower_left_y
            data_array.x_geostationary[-1].item(),  # upper_right_x
            data_array.y_geostationary[0].item(),  # upper_right_y
        ),
    )
    area_def = pyresample.create_area_def(
        area_id="msg_seviri_rss_1km",
        projection="WGS84",
        shape=(height, width),
        area_extent=cfg.lower_left_coords + cfg.upper_right_coords,
    )

    # Launch projection
    projected_xr = xr.apply_ufunc(
        reproject,
        data_array.chunk(
            dict(time_utc=1, channel=cfg.chunks, y_geostationary=-1, x_geostationary=-1)
        ),
        input_core_dims=[["y_geostationary", "x_geostationary"]],
        exclude_dims=set(["y_geostationary", "x_geostationary"]),
        output_core_dims=[["latitude", "longitude"]],
        output_sizes={"latitude": height, "longitude": width},
        dask="parallelized",
        kwargs={"area_def": area_def, "geos_area_def": geos_area_def},
        output_dtypes=[np.float16],
    )
    projected_xr = projected_xr.assign_coords(latitude=lat, longitude=lon)
    projected_xr = projected_xr.sortby("time_utc")

    start = time.time()
    projected_xr.to_dataset(name="data").to_zarr(Path(cfg.save_path), mode="w")
    print("Projection took", time.time() - start, "seconds.")


if __name__ == "__main__":
    main()
