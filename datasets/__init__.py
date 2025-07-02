"""
__init__ file for data classes.
imports all the data classes and creates a function to map the
    dataset name to the data class.
"""


from typing import Any, Tuple, List

from yacs.config import CfgNode

from datasets.vision import load_cifar10
from datasets.loaders import get_loaders


def get_dataset(dataset_name: str, cfg: CfgNode) -> Tuple[Any, List[Tuple[str, Any]]]:
    """
    Function to map dataset name to dataset class.
    Args:
        dataset_name (str): name of the dataset used for the experiment.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        dataset_objs (Tuple[Any, List[Tuple[str, Any]]]): dataset objects.
    """

    formatted_dataset_name = dataset_name.lower()

    if formatted_dataset_name == "cifar10":
        dataset_class = load_cifar10
    else:
        raise ValueError(f"Argument `dataset_name` not recognized (got `{dataset_name}`).")

    dataset_objs = dataset_class(cfg)

    return dataset_objs