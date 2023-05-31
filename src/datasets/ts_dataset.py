import json
from typing import Dict, List, Tuple, Union

import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.datasets.utils import calculate_possible_starts

"""
This prevents the following error occuring from the interaction between deeplake and wandb:
wandb.errors.UsageError: problem
"""
deeplake.constants.WANDB_INTEGRATION_ENABLED = False


class TS_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stats_path: str,
        ts_channels: Union[Tuple[str], List[str]],
        ts_target_channels: Union[Tuple[str], List[str]],
        years: Dict[str, Union[int, str]],
        stations: Dict[str, str],
        mode: str = "train",
        use_target: bool = True,
        seq_len: int = 24 * 2,
        label_len: int = 12,
        pred_len: int = 24 * 2,
    ) -> None:

        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        stats_path : str
            Absolute directory path where the stats json file is located.
        context_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the context as input. If ``None`` is given,
            then it takes all available channels.
        ts_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as input. If ``None`` is given,
            then it takes all available channels.
        ts_target_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as output. If ``None`` is given,
            then it takes all available channels.
        years: Dict[str, Union[int, str]]
            Dictionary containing for each mode, which years to select.
        stations: Dict[str, str]
            Dictionary containing for each mode, which stations to select.
        mode : str
            Indicates which dataset is it: train, val or test.
        use_target: bool
            Indicates whether to output target or not. In case it's not output, the input timeseries will be 2*n_steps long.
        image_size : Tuple[int]
            Interpolate to desired image size. If set to None, no interpolation is done
            and the size is kept as is.
        crop: Tuple[int]
            If not None, ``crop`` is expected to be in the following format:
                (lat_upper_left, lon_upper_left, lat_bottom_right, lon_bottom_right)
            And the context is cropped to the given parameters. Note that the context is first resized
            using ``image_size`` argument and then cropped.
        seq_len : int
            Length of the input sequence.
        label_len : int
            Length of the label sequence.
        pred_len : int
            Length of the prediction sequence.
        """


        self.data_dir = data_dir
        self.stats_path = stats_path
        self.ts_channels = ts_channels
        self.ts_target_channels = ts_target_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.mode = mode
        self.use_target = use_target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.n_samples = []
        self.year_mapping = {}

        if self.stats_path is not None:
            with open(self.stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None
        self.deeplake_ds = deeplake.load(self.data_dir, read_only=True)

        for year in self.years:
            for station in self.stations:
                ds = self.deeplake_ds[f"{year}/{station}"]
                _, possible_starts = calculate_possible_starts(
                    self.deeplake_ds[f"{year}/context/time_utc"].numpy()[0],
                    ds["time_utc"].numpy()[0],
                    frames_total=self.seq_len + self.pred_len,
                )
                n_in_group = len(possible_starts)
                self.n_samples.append(n_in_group)

                for i in range(self.n_samples[-1]):
                    step_id = possible_starts[i]
                    self.year_mapping[i + sum(self.n_samples[:-1])] = (
                        str(year),
                        station,
                        step_id,
                    )
        # Lazy loading
        self._mean = {}
        self._std = {}
        self._ts_channel_ids = None
        self._ts_target_channel_ids = None
        self._elevation = None

    def get_stats(self, station):
        if station in self._mean:
            return self._mean[station], self._std[station]

        if self.stats is not None:
            mean = []
            std = []
            station_channels = (
                self.ts_channels
                if self.ts_channels is not None
                else self.stats[station]
            )
            for i, chan in enumerate(station_channels):
                mean.append(float(self.stats[station][chan]["mean"]))
                std.append(float(self.stats[station][chan]["std"]))
            mean = torch.tensor(mean).float().view(1, -1)
            std = torch.tensor(std).float().view(1, -1)
            self._mean[station] = mean
            self._std[station] = std
            return mean, std
        return None, None

    def get_elevation(self, year):
        if self._elevation is None:
            self._elevation = self.deeplake_ds[f"{str(year)}/context/elevation"].numpy()
        return self._elevation

    def get_channel_ids(self, timeseries_tensor: deeplake.Tensor):
        """
        Get the list of channel indices to use for timeseries.
        Args:
            timeseries_tensor (deeplake.Tensor): Timeseries tensor to extract channel ids from.
        """

        if self._ts_channel_ids is None:
            if self.ts_channels is not None:
                self._ts_channel_ids = [
                    i
                    for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
                    for c in self.ts_channels
                    if c == k
                ]
            else:
                self._ts_channel_ids = [
                    i
                    for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
                ]
            self._ts_channel_ids = sorted(self._ts_channel_ids)

        return self._ts_channel_ids

    def get_target_channel_ids(self, ts_tensor):
        if self._ts_target_channel_ids is None:
            if self.ts_target_channels is not None:
                self._ts_target_channel_ids = [
                    i
                    for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                    for c in self.ts_target_channels
                    if c == k
                ]
            else:
                self._ts_target_channel_ids = [
                    i for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                ]
            self._ts_target_channel_ids = sorted(self._ts_target_channel_ids)
        return self._ts_target_channel_ids

    def __len__(self) -> int:
        return sum(self.n_samples)

    def __getitem__(self, idx: int) -> dict:
        year, station, step_idx = self.year_mapping[idx]

        x_begin_index = step_idx
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        if self.use_target:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0, x_begin_index : x_end_index
                ].numpy()
            )
        else:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0, x_begin_index : x_end_index + self.seq_len
                ].numpy()
            )

        mean, std = self.get_stats(station)

        ts_tensor = self.deeplake_ds[f"{year}/{station}/data"]
        ts_channel_ids = self.get_channel_ids(ts_tensor)
        target_channel_ids = self.get_target_channel_ids(ts_tensor)
        if self.use_target:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index : x_end_index,
                    ts_channel_ids,
                ].numpy()
            )
        else:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index : x_end_index + self.seq_len,
                    ts_channel_ids,
                ].numpy()
            )
        if mean is not None:
            timseries_data = (timseries_data - mean) / std

        if self.use_target:
            target = torch.from_numpy(
                ts_tensor[
                    0,
                    y_begin_index : y_end_index,
                    target_channel_ids,
                ].numpy()
            )
            target_previous = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index : x_end_index,
                    target_channel_ids,
                ].numpy()
            )

        elevation = torch.from_numpy(self.get_elevation(year))
        station_elevation = (
            torch.Tensor(
                [
                    ts_tensor.info["elevation"],
                ]
            )
            - elevation.mean()
        ) / elevation.std()
        station_elevation = station_elevation.unsqueeze(0)
        station_coords = [
            2 * (ts_tensor.info["coordinates"][0] + 90) / 180 - 1,
            2 * (ts_tensor.info["coordinates"][1] + 180) / 360,
        ]
        station_coords = torch.Tensor(station_coords)[(...,) + (None,) * 2]

        months = torch.from_numpy(time_utc.dt.month.values).unsqueeze(1)
        days = torch.from_numpy(time_utc.dt.day.values).unsqueeze(1)
        hours = torch.from_numpy(time_utc.dt.hour.values).unsqueeze(1)
        minutes = torch.from_numpy(time_utc.dt.minute.values).unsqueeze(1)

        time = torch.cat([months, days, hours, minutes], dim=1)
        return_tensors = {
            "timeseries": timseries_data,
            "time_coordinates": time,
            "station_elevation": station_elevation,
            "station_coords": station_coords,
        }
        if self.use_target:
            return_tensors["target"] = target
            return_tensors["target_previous"] = target_previous
        return return_tensors
