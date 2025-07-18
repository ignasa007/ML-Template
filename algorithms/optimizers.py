from typing import Iterator
from yacs.config import CfgNode

from torch.nn import Parameter
import torch.optim as optim


def get_optimizer(parameters: Iterator[Parameter], cfg: CfgNode) -> optim.Optimizer:
    """
    Function to map optimizer name to optimizer class.
    Args:
        cfg (yacs.CfgNode): configurations picked from ./config/.
        parameters (Iterator[Parameter]): model parameters to optimize.
    Returns:
        optimizer_class (torch.optim.Optimizer): optimization algorithm.
    """

    formatted_optimizer_name = cfg.optimizer.name.lower()

    if formatted_optimizer_name in ("none", "null"):
        print(f"Received `cfg.optimizer.name = {cfg.optimizer.name}`. Defaulting to SGD.")
        optimizer_class = optim.SGD
    elif formatted_optimizer_name in ("gd", "sgd"):
        optimizer_class = optim.SGD
    elif formatted_optimizer_name == "rmsprop":
        optimizer_class = optim.RMSprop
    elif formatted_optimizer_name == "adam":
        optimizer_class = optim.Adam
    elif formatted_optimizer_name == "adamw":
        optimizer_class = optim.AdamW
    else:
        raise ValueError(f"Argument `cfg.optimizer.name` not recognized (got `{cfg.optimizer.name}`).")

    optimizer_obj = optimizer_class(parameters, **cfg.optimizer.kwargs)

    return optimizer_obj