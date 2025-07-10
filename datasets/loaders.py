from typing import List

from yacs.config import CfgNode
import torch
from torch.utils.data import DataLoader

from datasets import BaseDataset


def get_loaders(
    *datasets: BaseDataset,
    cfg: CfgNode,
    device: torch.device
) -> List[DataLoader]:

    """"
    - If the target device is CPU, move the dataset to CPU.
    - If the target device is GPU,
        - move the dataset to GPU if it is small enough to be accommodated
        - keep the dataset on CPU if it is too large, but modify the target device for samples
    - Use `cfg.loader.pin_memory` only if data is stored on CPU but target device is GPU, else set it to False

    Args:
        datasets (List[datasets.BaseDataset]): list of datasets, e.g. train, val, test.
        cfg (yacs.CfgNode): experiment configurations.
        device (torch.device): target device for requested samples
    Returns:
        data_loaders (List[torch.utils.data.DataLoader]): list of data-loaders.
    """

    data_loaders = list()

    for dataset in datasets:

        if device.type == "cpu":
            dataset.to(device)
        elif device.type == "cuda":
            if cfg.dataset.to_cuda:
                # Push to GPU if VRAM would permit; user has to specify
                dataset.to(device)
        else:
            raise ValueError(f"Unrecognized `device.type={device.type}`.")
        
        # Note the device to be used by __getitem__ and collate methods
        dataset.target_device = device

        # Set pin_memory only if data is on CPU but target device is GPU
        kwargs = dict(cfg.loader)
        kwargs["pin_memory"] = dataset.storage_device.type == "cpu" \
            and dataset.target_device.type == "cuda" \
            and cfg.loader.pin_memory

        data_loaders.append(DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            collate_fn=dataset.collate,
            **kwargs
        ))

    return data_loaders