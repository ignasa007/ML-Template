from typing import Any, Tuple, Dict

from yacs.config import CfgNode
from torchvision import transforms


def get_transform(transform: Dict[str, Any]) -> Any:
    """
    Function to map transform name to transform obj.
    Args:
        transform_name (Dict[str, Any]): name and kwargs of the transform.
    Returns:
        transform_obj (Any): a transform class.
    """

    kwargs = transform.copy()
    formatted_transform_name = str(kwargs.pop("name", None)).lower()

    if formatted_transform_name in ("none", "null"):
        print(f"Received `{transform['name'] = }`. Skipping.")
    elif formatted_transform_name == "totensor":
        transform_class = transforms.ToTensor
    else:
        raise ValueError(f"Argument `transform['name']` not recognized (got `{transform['name']}`).")

    transform_obj = transform_class(**kwargs)

    return transform_obj


def compose(cfg: CfgNode, train: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Args:
        cfg (yacs.CfgNode): experiment configurations.
        train (bool): Boolean indicating whether transforming training or evaluation set.
    Returns:
        input_transforms (transforms.Compose): sequence of input transforms composed.
        target_transforms (transforms.Compose): sequence of target transforms composed.
    """

    if train:
        split_transforms = cfg.dataset.transforms.train
    else:
        split_transforms = cfg.dataset.transforms.eval
    
    input_transforms = transforms.Compose(list(map(get_transform, split_transforms.input)))
    target_transforms = transforms.Compose(list(map(get_transform, split_transforms.target)))

    return input_transforms, target_transforms