import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import pathlib
from pathlib import Path
from typing import List, Tuple, Union

import dask
import deeplake
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

dask.config.set(**{"array.slicing.split_large_chunks": True})


def save_context(
    deeplake_dataset: deeplake.Dataset,
    context_dir: pathlib.PosixPath,
    elevation_path: pathlib.PosixPath,
    image_size: Tuple[int],
    batch_size: int = 16,
) -> deeplake.Dataset:
    """
    Saves context data from a directory of zarr files to a deeplake dataset.

    Args:
        deeplake_dataset (deeplake.Dataset): A deeplake dataset to save the context data to.
        context_dir (pathlib.PosixPath): A directory containing zarr files for the context.
        elevation_path (pathlib.PosixPath): A file path to an elevation numpy array.
        image_size (Tuple[int]): A tuple containing the height and width to interpolate the contextual data to.
        batch_size (int, optional): The batch size for loading the context data. Defaults to 16.

    Returns:
        deeplake.Dataset: A deeplake dataset containing the context data.
    """
    elevation = np.load(elevation_path)
    years = []
    for path in tqdm(context_dir.glob("*.zarr")):
        year = path.stem
        years.append(year)
        deeplake_dataset.create_group("/".join([year, "context"]), exist_ok=True)
        with deeplake_dataset:
            context_grp = deeplake_dataset["/".join([year, "context"])]
            context_grp.create_tensor(
                "data",
                sample_compression="lz4",
                dtype=np.float32,
            )
            context_grp.create_tensor(
                "latitude",
                sample_compression="lz4",
                dtype=np.float32,
            )
            context_grp.create_tensor(
                "elevation",
                sample_compression="lz4",
                dtype=np.float32,
            )
            context_grp.create_tensor(
                "longitude",
                sample_compression="lz4",
                dtype=np.float32,
            )
            context_grp.create_tensor(
                "time_utc",
                sample_compression="lz4",
                dtype=np.dtype("<M8[ns]"),
            )

            xr_dataset = (
                xr.open_dataset(
                    path,
                    engine="zarr",
                    chunks="auto",
                )
                .drop_duplicates("time_utc")
                .sortby("time_utc")
            )
            xr_dataset = xr_dataset.dropna(dim="time_utc", how="any")

            T, C, H, W = xr_dataset.data.shape
            lat = torch.from_numpy(xr_dataset.latitude.values[::-1].astype(np.float32))
            lat = lat.unsqueeze(0).unsqueeze(0)
            lon = torch.from_numpy(xr_dataset.longitude.values.astype(np.float32))
            lon = lon.unsqueeze(0).unsqueeze(0)
            elevation_ = torch.from_numpy(elevation["arr_0"].astype(np.float32))
            elevation_ = elevation_.unsqueeze(0).unsqueeze(0)
            if image_size != (H, W):
                lat = (
                    F.interpolate(lat, size=(image_size[0],), mode="linear")
                    .squeeze()
                    .numpy()
                )
                lon = (
                    F.interpolate(lon, size=(image_size[1],), mode="linear")
                    .squeeze()
                    .numpy()
                )
                elevation_ = (
                    F.interpolate(elevation_, size=image_size, mode="bilinear")
                    .squeeze()
                    .numpy()
                )
                xr_dataset = xr_dataset.interp(
                    coords={"latitude": lat, "longitude": lon}, method="linear"
                )
            else:
                lat = lat.squeeze().numpy()
                lon = lon.squeeze().numpy()
                elevation_ = elevation_.squeeze().numpy()

            context_grp["latitude"].extend(lat)
            context_grp["longitude"].extend(lon)
            context_grp["elevation"].extend(elevation_)
            context_grp["time_utc"].append(xr_dataset.time_utc.values)

            for i in tqdm(range((T // batch_size) + 1), leave=False):
                x = (
                    xr_dataset.data[batch_size * i : batch_size * (i + 1)]
                    .load()
                    .values.astype(np.float32)
                )
                context_grp["data"].extend(x)

            context_grp["data"].info[
                "context_channels"
            ] = xr_dataset.channel.values.tolist()
    return deeplake_dataset


def save_timeseries(
    deeplake_dataset: deeplake.Dataset,
    timeseries_paths: List[pathlib.PosixPath],
    ts_elevations: List[float],
    ts_coordinates: List[Union[Tuple[float], List[float]]],
) -> deeplake.Dataset:
    """
    Parses timeseries from csv and puts them inside the deeplake dataset.
    Arguments:
        deeplake_dataset (deeplake.Dataset): A deeplake.Dataset object.
        timeseries_paths (List[pathlib.PosixPath]): A list of file paths of the timeseries data to be parsed and saved in ``deeplake_dataset``.
        ts_elevations (List[float]): List of elevations of the timeseries data.
        ts_coordinates (List[Union[Tuple[float], List[float]]]): List of coordinates of the timeseries data.
    Returns:
        deeplake_dataset (deeplake.Dataset): A deeplake dataset with the parsed timeseries data saved.
    """

    for (ts_path, elevation, coords) in tqdm(
        zip(timeseries_paths, ts_elevations, ts_coordinates)
    ):
        df = pd.read_csv(ts_path)
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], infer_datetime_format=True)
        if "Unnamed: 0" in df:
            df = df.drop("Unnamed: 0", axis=1)

        for year in deeplake_dataset.groups:
            deeplake_dataset.create_group("/".join([year, ts_path.stem]), exist_ok=True)
            with deeplake_dataset:
                timeseries_grp = deeplake_dataset["/".join([year, ts_path.stem])]
                df_ = df[df["GHI"].notna()]
                df_.drop_duplicates("Date/Time", inplace=True)
                df_.sort_values(by="Date/Time", inplace=True)
                df_ = df_[df_["Date/Time"].dt.year == int(year.split("_")[0])]
                times = df_["Date/Time"].values
                df_ = df_.loc[:, df_.columns != "Date/Time"]

                timeseries_grp.create_tensor(
                    "data",
                    sample_compression="lz4",
                    dtype=np.float32,
                )
                timeseries_grp["data"].append(df_.values.astype(np.float32))

                timeseries_grp.create_tensor(
                    "time_utc",
                    sample_compression="lz4",
                    dtype=np.dtype("<M8[ns]"),
                )
                timeseries_grp["time_utc"].append(times)

                # Add some metadata
                timeseries_grp["data"].info[
                    "timeseries_channels"
                ] = df_.columns.values.tolist()
                timeseries_grp["data"].info["elevation"] = elevation
                timeseries_grp["data"].info["coordinates"] = coords
    return deeplake_dataset


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="to_deeplake.yaml"
)
def main(cfg: DictConfig) -> None:

    if isinstance(cfg.image_size, int):
        image_size = (cfg.image_size, cfg.image_size)
    else:
        image_size = tuple(cfg.image_size)

    save_dir = Path("-".join([cfg.save_dir, "_".join(map(str, image_size))]))
    context_dir = Path(cfg.context_dir)
    timeseries_paths = [Path(path) for path in cfg.timeseries_paths]
    elevation_path = Path(cfg.elevation_path)
    batch_size = cfg.batch_size
    ts_elevations = cfg.ts_elevations
    ts_coordinates = OmegaConf.to_object(cfg.ts_coordinates)

    ds = deeplake.dataset(save_dir)
    ds = save_context(ds, context_dir, elevation_path, image_size, batch_size)
    ds = save_timeseries(ds, timeseries_paths, ts_elevations, ts_coordinates)


if __name__ == "__main__":
    main()
