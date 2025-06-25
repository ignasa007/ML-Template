from typing import Tuple

from yacs.config import CfgNode
import torch.nn as nn
from torchvision.transforms import Compose


map = {
    # transform name: transform class
}

def get_transform(transform_name: str, cfg: CfgNode) -> nn.Module:
    """
    Function to map transform name to transform class.
    Args:
        transform_name (str): name of the transform used for the experiment.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        transform_class (nn.Module): a transform class.
    """

    formatted_transform_name = transform_name.lower()
    if formatted_transform_name not in map:
        raise ValueError(
            "Parameter `transform_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_transform_name},' for valid_transform_name in map.keys()) +
            f"\nbut got `{transform_name}`."
        )

    transform_class = map.get(formatted_transform_name)

    return transform_class(cfg)


def compose(cfg: CfgNode, train: bool) -> Tuple[Compose, Compose]:
    """
    Function to compose transforms; different logic for train/eval.
    If `train=True`, then in addition to common transforms, eg. some processing,
        also apply training-specific transforms, eg. random crop, label smoothing.
    Args:
        cfg (yacs.CfgNode): experiment configurations.
        train (bool): Boolean indicating whether transforming training or evaluation set.
    Returns:
        transform_class (nn.Module): a transform class.
    """

    input_transforms = list()
    for transform_name in cfg.dataset.common_transforms.input:
        input_transforms.append(get_transform(transform_name))

    target_transforms = list()
    for transform_name in cfg.dataset.common_transforms.target:
        target_transforms.append(get_transform(transform_name))

    if train:
        for transform_name in cfg.dataset.train_transforms.input:
            input_transforms.append(get_transform(transform_name))
        for transform_name in cfg.dataset.train_transforms.target:
            target_transforms.append(get_transform(transform_name))

    return Compose(input_transforms), Compose(target_transforms)