"""
__init__ file for data classes.
imports all the data classes and creates a function to map the 
    dataset name to the data class.
"""

from typing import Tuple, List

from yacs.config import CfgNode

from datasets.base import BaseDataset
from datasets.vision import load_cifar10
from datasets.loaders import get_loaders


map = {
    # data name: data class
    'cifar10': load_cifar10
}

def get_dataset(
    dataset_name: str,
    cfg: CfgNode
) -> Tuple[BaseDataset, List[Tuple[str, BaseDataset]]]:
    """
    Function to map dataset name to dataset class.
    Args:
        dataset_name (str): name of the dataset used for the experiment.
        cfg (yacs.CfgNode): experiment configurations.
    Return:
        dataset_class (torch.data.BaseDataset): a data class.
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