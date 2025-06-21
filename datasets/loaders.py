from typing import List

from yacs.config import CfgNode
from torch import device as Device
from torch.utils.data import DataLoader

from datasets.base import BaseDataset


def get_loaders(
    *datasets: List[BaseDataset],
    cfg: CfgNode,
    device: Device
) -> List[DataLoader]:

    """"
    - If the target device is CPU, move the dataset to CPU.
    - If the target device is GPU,
        - move the dataset to GPU if it is small enough to be accommodated
        - keep the dataset on CPU if it is too large, but modify the target device for samples
    - Use `cfg.loader.pin_memory` only if data is stored on CPU but target device is GPU, else set it to False

    Args:
        datasets (List[data.base.BaseDataset, ...]): list of datasets, e.g. train, val, test.
        cfg (yacs.CfgNode): experiment configurations.
        device (torch.device): target device for requested samples
    
    Return:
        data_loaders (List[torch.utils.data.DataLoader, ...]): list of data-loaders.
    """

    data_loaders = list()
    
    for dataset in datasets:
        
        if device.type == "cpu":
            dataset.to(device)
        elif device.type == "cuda":
            if cfg.dataset.to_cuda:
                dataset.to(device)
            # Note the target device to be used in the __getitem__ method
            else:
                dataset.target_device = device
        else:
            raise ValueError(f"Unrecognized `device.type = {device.type}`.")

        # Set pin_memory only if data is on CPU and target device is GPU
        kwargs = dict(cfg.loader)
        kwargs["pin_memory"] = dataset.storage_device.type == "cpu" \
            and dataset.target_device.type == "cuda" \
            and cfg.laoder.pin_memory
            
        data_loaders.append(DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            collate_fn=dataset.collater,
            **kwargs
        ))

    return data_loaders