from typing import Tuple, List, Callable

from yacs.config import CfgNode
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """For type checking."""

    storage_device = torch.device("cpu")
    target_device = torch.device("cpu")

    def to(self, device: torch.device):
        """Transfer the samples to `device`."""
        return self


_DATASET_REGISTRY = dict()

def register_dataset(*dataset_names: str) -> Callable:
    
    def decorator(cls):
        for dataset_name in dataset_names:
            if dataset_name in _DATASET_REGISTRY:
                raise ValueError(f"`dataset_name={dataset_name}` already registered.")
            _DATASET_REGISTRY[dataset_name] = cls
        return cls
    
    return decorator

def get_dataset(dataset_name: str, cfg: CfgNode) -> Tuple[Tuple, Dataset, List[Tuple[str, Dataset]]]:
    """
    Function to map dataset name to dataset class.
    Args:
        dataset_name (str): name of the dataset used for the experiment.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        input_dim (tuple): model input dimension (in case not using LazyInit).
        training_dataset (Any): training dataset.
        evaluation_datasets (List[Tuple[str, Any]]): evaluation datasets.
    """
    
    formatted_dataset_name = dataset_name.lower()
    if formatted_dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Argument `dataset_name` not recognized (got `{dataset_name}`).")
    
    dataset_class = _DATASET_REGISTRY[formatted_dataset_name]
    input_dim, training_dataset, evaluation_datasets = dataset_class(cfg)
    
    return input_dim, training_dataset, evaluation_datasets

def list_datasets() -> List:
    
    return list(_DATASET_REGISTRY.keys())