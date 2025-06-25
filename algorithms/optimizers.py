from typing import Iterator
from yacs.config import CfgNode

from torch.nn import Parameter
from torch.optim import (
    Optimizer,
    SGD,
    RMSprop,
    Adam,
    AdamW,
)

map = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adam': Adam,
    'adamw': AdamW,
}

def get_optimizer(parameters: Iterator[Parameter], cfg: CfgNode) -> Optimizer:
    """
    Function to map optimizer name to optimizer class.
    Args:
        cfg (yacs.CfgNode): configurations picked from ./config/.
        parameters (dict): model parameters to optimize.
    Returns:
        optimizer_class (torch.optim.Optimizer): optimization algorithm.
    """

    formatted_optimizer_name = cfg.optimizer.name.lower()
    if formatted_optimizer_name not in map:
        raise ValueError(
            "Parameter `optimizer.name` not recognized. Expected one of" +
            "".join(f'\n\t- {name},' for name in map.keys()) +
            f"\nbut got `{cfg.optimizer.name}`."
        )

    optimizer_class = map.get(formatted_optimizer_name)

    return optimizer_class(parameters, **dict(cfg.optimizer))