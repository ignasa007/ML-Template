from yacs.config import CfgNode
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    ConstantLR,
    StepLR,
    MultiStepLR,
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

map = {
    'constant': ConstantLR,
    'step': StepLR,
    'multistep': MultiStepLR,
    'linear': LinearLR,
    'exponential': ExponentialLR,
    'cosine': CosineAnnealingLR,
    'cosine_restart': CosineAnnealingWarmRestarts,
}

def get_scheduler(optimizer: Optimizer, cfg: CfgNode) -> LRScheduler:
    """
    Function to map scheduler name to scheduler class.
    Args:
        cfg (yacs.CfgNode): configurations picked from ./config/.
        optimizer (torch.optim.Optimizer): optimizer instance to schedule.
    Returns:
        scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler.
    """

    formatted_name = cfg.scheduler.name.lower()
    if formatted_name not in map:
        raise ValueError(
            "Parameter `scheduler.name` not recognized. Expected one of" +
            "".join(f'\n\t- {name}' for name in map.keys()) +
            f"\nbut got `{cfg.scheduler.name}`."
        )

    scheduler_class = map.get(formatted_name)

    return scheduler_class(optimizer, **dict(cfg.scheduler))