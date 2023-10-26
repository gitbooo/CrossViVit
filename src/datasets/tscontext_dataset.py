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


class TSContextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stats_path: str,
        context_channels: Union[Tuple[str], List[str]],
        optflow_channels: Union[Tuple[str], List[str]],
        ts_channels: Union[Tuple[str], List[str]],
        ts_target_channels: Union[Tuple[str], List[str]],
        years: Dict[str, Union[int, str]],
        stations: Dict[str, str],
        mode: str = "train",
        use_target: bool = True,
        image_size: Tuple[int] = None,
        crop: Tuple[int] = None,
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
        optflow_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the optical flow as input. If ``None`` is given,
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
        seq_len: int
            Number of frames in the input sequence.
        label_len: int
            Number of frames in the label sequence.
        pred_len: int
            Number of frames in the prediction sequence.
        """

        self.data_dir = data_dir
        self.stats_path = stats_path
        self.context_channels = context_channels
        self.optflow_channels = optflow_channels
        self.ts_channels = ts_channels
        self.ts_target_channels = ts_target_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.crop = crop
        self.mode = mode
        self.image_size = tuple(image_size) if image_size is not None else None
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
                (
                    possible_starts_context,
                    possible_starts_station,
                ) = calculate_possible_starts(
                    self.deeplake_ds[f"{year}/context/time_utc"].numpy()[0],
                    self.deeplake_ds[f"{year}/{station}/time_utc"].numpy()[0],
                    frames_total=self.seq_len + self.pred_len,
                )
                n_in_group = len(possible_starts_context)
                self.n_samples.append(n_in_group)

                for i in range(self.n_samples[-1]):
                    step_id_station = possible_starts_station[i]
                    step_id_context = possible_starts_context[i]
                    self.year_mapping[i + sum(self.n_samples[:-1])] = (
                        str(year),
                        station,
                        step_id_station,
                        step_id_context,
                    )
        # Lazy loading
        self._coords = None
        self._elevation = None
        self._mean = {}
        self._std = {}
        self._context_channel_ids = None
        self._optflow_channel_ids = None
        self._ts_channel_ids = None
        self._ts_target_channel_ids = None
        self._lat_slice = None
        self._lon_slice = None

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

    def get_coords(self, year):
        if self._coords is None:
            lat = self.deeplake_ds[f"{str(year)}/context/latitude"].numpy()

            lat = 2 * ((lat + 90) / 180) - 1
            lon = self.deeplake_ds[f"{str(year)}/context/longitude"].numpy()
            lon = 2 * ((lon + 180) / 360) - 1
            self._coords = np.stack(np.meshgrid(lat, lon), axis=0)
        return self._coords

    def get_elevation(self, year):
        if self._elevation is None:
            self._elevation = self.deeplake_ds[f"{str(year)}/context/elevation"].numpy()
        return self._elevation

    def get_channel_ids(
        self,
        context_tensor: deeplake.Tensor,
        optflow_tensor: deeplake.Tensor,
        timeseries_tensor: deeplake.Tensor,
    ):
        """
        Get the list of channel indices to use for the context, optical flow and timeseries.
        Args:
            context_tensor (deeplake.Tensor): Context tensor to extract channel ids from.
            optflow_tensor (deeplake.Tensor): Optical flow tensor to extract channel ids from.
            timeseries_tensor (deeplake.Tensor): Timeseries tensor to extract channel ids from.
        """
        if self._context_channel_ids is None:
            if self.context_channels is not None:
                self._context_channel_ids = [
                    i
                    for i, k in enumerate(context_tensor.info["context_channels"])
                    for c in self.context_channels
                    if c == k
                ]
            else:
                self._context_channel_ids = [
                    i for i, k in enumerate(context_tensor.info["context_channels"])
                ]
            self._context_channel_ids = sorted(self._context_channel_ids)

        if self._optflow_channel_ids is None:
            if self.optflow_channels is not None:
                self._optflow_channel_ids = [
                    i
                    for i, k in enumerate(optflow_tensor.info["optflow_channels"])
                    for c in self.optflow_channels
                    if c == k
                ]
            else:
                self._optflow_channel_ids = [
                    i for i, k in enumerate(optflow_tensor.info["optflow_channels"])
                ]
            self._optflow_channel_ids = sorted(self._optflow_channel_ids)

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

        return (
            self._context_channel_ids,
            self._optflow_channel_ids,
            self._ts_channel_ids,
        )

    def get_target_channel_ids(self, ts_tensor):
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
        year, station, step_idx_station, step_idx_context = self.year_mapping[idx]

        x_begin_index_ctx = step_idx_context
        x_end_index_ctx = x_begin_index_ctx + self.seq_len

        x_begin_index_ts = step_idx_station
        x_end_index_ts = x_begin_index_ts + self.seq_len
        y_begin_index_ts = x_end_index_ts - self.label_len
        y_end_index_ts = y_begin_index_ts + self.label_len + self.pred_len

        if self.use_target:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0,
                    x_begin_index_ts:x_end_index_ts,
                ].numpy()
            )
        else:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0,
                    x_begin_index_ts : x_end_index_ts + self.seq_len,
                ].numpy()
            )

        mean, std = self.get_stats(station)

        ts_tensor = self.deeplake_ds[f"{year}/{station}/data"]
        context_tensor = self.deeplake_ds[f"{year}/context/data"]
        optflow_tensor = self.deeplake_ds[f"{year}/ctx_opt_flow/data"]
        context_channel_ids, optflow_channel_ids, ts_channel_ids = self.get_channel_ids(
            context_tensor, optflow_tensor, ts_tensor
        )
        target_channel_ids = self.get_target_channel_ids(ts_tensor)
        if self.use_target:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index_ts:x_end_index_ts,
                    ts_channel_ids,
                ].numpy()
            )
        else:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index_ts : x_end_index_ts + self.seq_len,
                    ts_channel_ids,
                ].numpy()
            )
        if mean is not None:
            timseries_data = (timseries_data - mean) / std
        if self.use_target:
            target = torch.from_numpy(
                ts_tensor[
                    0,
                    y_begin_index_ts:y_end_index_ts,
                    target_channel_ids,
                ].numpy()
            )
            target_previous = torch.from_numpy(
                ts_tensor[
                    0, x_begin_index_ts:x_end_index_ts, target_channel_ids
                ].numpy()
            )
        if self.use_target:
            context_data = torch.from_numpy(
                context_tensor[
                    x_begin_index_ctx:x_end_index_ctx,
                    context_channel_ids,
                ].numpy()
            )
            optflow_data = torch.from_numpy(
                optflow_tensor[
                    x_begin_index_ctx:x_end_index_ctx,
                    optflow_channel_ids,
                ].numpy()
            )
        else:
            context_data = torch.from_numpy(
                context_tensor[
                    x_begin_index_ctx : x_end_index_ctx + self.seq_len,
                    context_channel_ids,
                ].numpy()
            )
            optflow_data = torch.from_numpy(
                optflow_tensor[
                    x_begin_index_ctx : x_end_index_ctx + self.seq_len,
                    optflow_channel_ids,
                ].numpy()
            )
        coords = torch.from_numpy(self.get_coords(year))
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
        elevation = (elevation - elevation.mean()) / elevation.std()

        if self.image_size is not None:
            optflow_data = F.interpolate(
                optflow_data, size=self.image_size, mode="bilinear", align_corners=True
            )
            context_data = F.interpolate(
                context_data, size=self.image_size, mode="bilinear", align_corners=True
            )
            coords = F.interpolate(
                coords.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            elevation = F.interpolate(
                elevation.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze()

        H, W = context_data.shape[-2:]

        months = torch.from_numpy(time_utc.dt.month.values)[
            (...,) + (None,) * 3
        ].repeat(1, 1, H, W)
        days = torch.from_numpy(time_utc.dt.day.values)[(...,) + (None,) * 3].repeat(
            1, 1, H, W
        )
        hours = torch.from_numpy(time_utc.dt.hour.values)[(...,) + (None,) * 3].repeat(
            1, 1, H, W
        )
        minutes = torch.from_numpy(time_utc.dt.minute.values)[
            (...,) + (None,) * 3
        ].repeat(1, 1, H, W)

        time = torch.cat([months, days, hours, minutes], dim=1)

        return_tensors = {
            "context": context_data,
            "optical_flow": optflow_data,
            "timeseries": timseries_data,
            "spatial_coordinates": coords,
            "time_coordinates": time,
            "station_elevation": station_elevation,
            "station_coords": station_coords,
            "auxiliary_data": {"elevation": elevation},
        }
        if self.use_target:
            return_tensors["target"] = target
            return_tensors["target_previous"] = target_previous
        return return_tensors
