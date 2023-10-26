import json
from typing import Dict, List, Tuple, Union

import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.datasets.utils import calculate_possible_starts
from src.utils.time_features import time_features

"""
This prevents the following error occuring from the interaction between deeplake and wandb:
wandb.errors.UsageError: problem
"""
deeplake.constants.WANDB_INTEGRATION_ENABLED = False


class TSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stats_path: str,
        ts_channels: Union[Tuple[str], List[str]],
        years: Dict[str, Union[int, str]],
        stations: Dict[str, str],
        mode: str = "train",
        seq_len: int = 24 * 2,
        label_len: int = 12,
        pred_len: int = 24 * 2,
        freq: str = "t",
        time_encoding: bool= True,
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
        years: Dict[str, Union[int, str]]
            Dictionary containing for each mode, which years to select.
        stations: Dict[str, str]
            Dictionary containing for each mode, which stations to select.
        mode : str
            Indicates which dataset is it: train, val or test.
        image_size : Tuple[int]
            Interpolate to desired image size. If set to None, no interpolation is done
            and the size is kept as is.
        crop: Tuple[int]
            If not None, ``crop`` is expected to be in the following format:
                (lat_upper_left, lon_upper_left, lat_bottom_right, lon_bottom_right)
            And the context is cropped to the given parameters. Note that the context is first resized
            using ``image_size`` argument and then cropped.
        n_steps: int
            Number of past frames used for temporal context and number of future frames to predict.
        frame_step: int
            Number of frames to jump. If set to 1, we take all frames, otherwise, we take every ``frame_step`` frame.
        """

        assert (
            seq_len > 1
        ), f"n_steps should be at least 2 but you provided n_steps={seq_len}"
        assert (
            pred_len > 0
        ), f"frame_step should be at least 1 but you provided frame_step={pred_len}"

        self.data_dir = data_dir
        self.stats_path = stats_path
        self.ts_channels = ts_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.mode = mode
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.freq = freq
        self.time_encoding = time_encoding

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
                    frames_total= self.seq_len + self.pred_len,
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
        self._ghi_channel = None
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

    def get_ghi_channel(self, ts_tensor):
        if self._ghi_channel is None:
            self._ghi_channel = [
                i
                for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                if k == "GHI"
            ]
        return self._ghi_channel[0]

    def __len__(self) -> int:
        return sum(self.n_samples)

    def __getitem__(self, idx: int) -> dict:
        year, station, step_idx = self.year_mapping[idx]

        x_begin_index = step_idx
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len
        
        x_time_utc = pd.DataFrame(
            self.deeplake_ds[f"{year}/{station}/time_utc"][
                0, x_begin_index:x_end_index
            ].numpy(), columns=["date"]
        )

        x_time_utc= time_features(x_time_utc, time_encoding=self.time_encoding, frequency=self.freq)

        y_time_utc = pd.DataFrame(
            self.deeplake_ds[f"{year}/{station}/time_utc"][
                0, y_begin_index:y_end_index
            ].numpy() , columns=["date"]
        )
        y_time_utc= time_features(y_time_utc, time_encoding=self.time_encoding, frequency=self.freq)

        mean, std = self.get_stats(station)

        ts_tensor = self.deeplake_ds[f"{year}/{station}/data"]
        ts_channel_ids = self.get_channel_ids(ts_tensor)
        ghi_channel_id = self.get_ghi_channel(ts_tensor)

        x_timseries_data = torch.from_numpy(
            ts_tensor[
                0,
                x_begin_index:x_end_index,
                ts_channel_ids,
            ].numpy()
        )
        if mean is not None:
            x_timseries_data = (x_timseries_data - mean) / std

        y_timseries_data = torch.from_numpy(
            ts_tensor[
                0,
                y_begin_index:y_end_index,
                ts_channel_ids,
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

        return_tensors = {
            "x": x_timseries_data,
            "y": y_timseries_data,
            "x_timestamp": x_time_utc,
            "y_timestamp": y_time_utc,
            "station_coords": station_coords,
            "station_elevation": station_elevation,
            "ghi": ghi_channel_id,
        }
        return return_tensors
