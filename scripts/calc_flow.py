import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import multiprocessing
import warnings
from itertools import repeat

import cv2
import deeplake
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

warnings.filterwarnings("ignore")


def optflow_batch_helper(args):
    return optflow_batch(*args)


def optflow_batch(
    indices,
    ctx_data,
    params,
):
    out = []
    for s, e in indices:
        flows = []
        for j, _ in enumerate(ctx_data.info["context_channels"]):
            prev, curr = ctx_data[s:e, j].numpy().astype(np.float32)
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(**params)
            flow = (
                optical_flow.calc(prev, curr, flow=None)
                .transpose(2, 0, 1)
                .astype(np.float32)
            )  # C, H, W

            flows.append(flow)
        flows = np.concatenate(flows, axis=0)
        flows = flows[(None, ...)]
        out.append((flows))

    return out


def calculate_optical_flow(year, ds, cfg):
    ctx_data = ds["/".join([str(year), "context", "data"])]

    indices = [(0, 2)] + [(i - 1, i + 1) for i in range(1, ctx_data.shape[0])]
    batch_idx = [
        indices[k : k + cfg.batch_size] for k in range(0, len(indices), cfg.batch_size)
    ]
    params = OmegaConf.to_container(cfg.params, resolve=True)

    with open(
        f"/network/scratch/o/oussama.boussif/TSEumetsat/opticals/{year}.npy", "wb"
    ) as f:
        with multiprocessing.Pool(cfg.cpus) as p:
            for flows in tqdm(
                p.imap(
                    optflow_batch_helper,
                    zip(batch_idx, repeat(ctx_data), repeat(params)),
                ),
                total=len(batch_idx),
            ):
                np.save(f, flows)


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="opt_flow.yaml"
)
def main(cfg: DictConfig) -> None:
    ds = deeplake.load(cfg.dataset_path)

    calculate_optical_flow(cfg.year, ds, cfg)


if __name__ == "__main__":
    main()
