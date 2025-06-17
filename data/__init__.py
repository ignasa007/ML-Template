"""
__init__ file for data classes.
imports all the data classes and creates a function to map the 
    dataset name to the data class.
"""

from typing import List

from yacs.config import CfgNode
from torch import device as Device
from torch.utils.data import DataLoader

from data.base import BaseDataset


map = {
    # data name: data class
}


def get_datasets(dataset_name: str, cfg: CfgNode) -> BaseDataset:

    """
    Function to map dataset name to dataset class.

    Args:
        dataset_name (str): name of the dataset used for the experiment
    
    Return:
        dataset_class (BaseDataset): a data class
    """
    
    formatted_dataset_name = dataset_name.lower()
    if formatted_dataset_name not in map:
        raise ValueError(
            "Parameter `dataset_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_dataset_name},' for valid_dataset_name in map.keys()) +
            f"\nbut got `{dataset_name}`."
        )
    
    dataset_class = map.get(formatted_dataset_name)
    
    return dataset_class(cfg)


def get_loaders(*datasets: List[BaseDataset], cfg: CfgNode, device: Device) -> List[DataLoader]:

    out = list()
    for dataset in datasets:
        # If the dataset is not on CPU or target device is not GPU, don't need to `pin_memory` in the loader
        kwargs = dict(cfg.loader)
        if not (dataset.device.type == 'cpu' and device.type == 'cuda'):
            kwargs["pin_memory"] = False
        out.append(DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            collate_fn=dataset.collater,
            **kwargs
        ))

    return out