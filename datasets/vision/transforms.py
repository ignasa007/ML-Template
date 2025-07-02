from typing import Any, Tuple, Dict

from attrdict import AttrDict
from yacs.config import CfgNode
import torchvision.transforms as Transforms


def get_transform(transform: Dict[str, str]) -> Any:
    """
    Function to map transform name to transform class.
    Args:
        transform_name (Dict): name and kwargs of the transform.
    Returns:
        transform_obj (Any): a transform class.
    """

    kwargs = transform.copy()
    formatted_transform_name = str(kwargs.pop("name", None)).lower()

    if formatted_transform_name == "none":
        print(f"Received `{transform['name'] = }`. Skipping.")
    elif formatted_transform_name == "totensor":
        transform_class = Transforms.ToTensor
    else:
        raise ValueError(f"Argument `transform['name']` not recognized (got `{transform['name']}`).")

    transform_obj = transform_class(**kwargs)

    return transform_obj


def compose(cfg: CfgNode, train: bool) -> Tuple[Transforms.Compose, Transforms.Compose]:
    """
    Function to compose transforms; different logic for train/eval.
    If `train=True`, then in addition to common transforms, eg. some processing,
        also apply training-specific transforms, eg. random crop, label smoothing.
    Args:
        cfg (yacs.CfgNode): experiment configurations.
        train (bool): Boolean indicating whether transforming training or evaluation set.
    Returns:
        transform_class (transforms.Compose): a transform class.
    """

    default = AttrDict({"input": list(), "train": list()})
    if train:
        transforms = cfg.dataset.transforms.get("train", default)
    else:
        transforms = cfg.dataset.transforms.get("eval", default)
    
    input_transforms = list(map(get_transform, transforms.input))
    target_transforms = list(map(get_transform, transforms.target))

    return Transforms.Compose(input_transforms), Transforms.Compose(target_transforms)