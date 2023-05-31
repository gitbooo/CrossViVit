import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import json

import deeplake
import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="calc_stats.yaml"
)
def main(cfg: DictConfig) -> None:

    ds = deeplake.load(cfg.deeplake_dataset_path, read_only=True, reset=True)

    years = map(str, cfg.years)
    station_vars = {}
    station_channels = {}
    for key in ds.groups:
        if any([year in key for year in years]):
            for station in ds[key].groups:
                if not station in ["context", "ctx_opt_flow"]:
                    ts_tensor = ds[f"{key}/{station}/data"]
                    channels = ts_tensor.info["timeseries_channels"]
                    station_channels[station] = channels
                    if not station in station_vars:
                        station_vars[station] = [ts_tensor[0].numpy()]
                    else:
                        station_vars[station].append(ts_tensor[0].numpy())

    stats_dict = {}
    for station in station_vars:
        stats_dict[station] = {}
        x = np.concatenate(station_vars[station], axis=0)
        mean = x.mean(0)
        std = x.std(0)
        for i, chan in enumerate(station_channels[station]):
            stats_dict[station][chan] = {}
            stats_dict[station][chan]["mean"] = str(mean[i])
            stats_dict[station][chan]["std"] = str(std[i])

    with open(cfg.save_path, "w") as fp:
        json.dump(stats_dict, fp)


if __name__ == "__main__":

    main()
