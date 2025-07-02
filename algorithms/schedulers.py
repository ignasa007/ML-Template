from yacs.config import CfgNode
from torch.optim import Optimizer
import torch.optim.lr_scheduler as scheduler


class NoSchedule(scheduler.LRScheuler):
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def get_scheduler(optimizer: Optimizer, cfg: CfgNode) -> scheduler.LRScheduler:
    """
    Function to map scheduler name to scheduler class.
    Args:
        cfg (yacs.CfgNode): configurations picked from ./config/.
        optimizer (torch.optim.Optimizer): optimizer instance to schedule.
    Returns:
        scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler.
    """

    kwargs = dict(cfg.scheduler)
    scheduler_name = str(kwargs.pop("name", None))
    formatted_scheduler_name = scheduler_name.lower()

    if formatted_scheduler_name == "none":
        print(f"Received `cfg.scheduler.name = {scheduler_name}`. Defaulting to NoSchedule.")
        scheduler_class = NoSchedule
    elif formatted_scheduler_name == "noschedule":
        scheduler_class = NoSchedule
    elif formatted_scheduler_name == "constant":
        scheduler_class = scheduler.ConstantLR
    elif formatted_scheduler_name == "step":
        scheduler_class = scheduler.StepLR
    elif formatted_scheduler_name == "multistep":
        scheduler_class = scheduler.MultiStepLR
    elif formatted_scheduler_name == "linear":
        scheduler_class = scheduler.LinearLR
    elif formatted_scheduler_name == "exponential":
        scheduler_class = scheduler.ExponentialLR
    elif formatted_scheduler_name == "cosine":
        scheduler_class = scheduler.CosineAnnealingLR
    elif formatted_scheduler_name == "cosine_restart":
        scheduler_class = scheduler.CosineAnnealingWarmRestarts
    else:
        raise ValueError(f"Argument `cfg.scheduler.name` not recognized (got `{scheduler_name}`).")

    scheduler_obj = scheduler_class(optimizer, **kwargs)

    return scheduler_obj