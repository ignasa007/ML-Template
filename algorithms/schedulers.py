from yacs.config import CfgNode
from torch.optim import Optimizer, lr_scheduler


class NoSchedule(lr_scheduler.LRScheduler):
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def get_scheduler(optimizer: Optimizer, cfg: CfgNode) -> lr_scheduler.LRScheduler:
    """
    Function to map scheduler name to scheduler class.
    Args:
        cfg (yacs.CfgNode): configurations picked from ./config/.
        optimizer (torch.optim.Optimizer): optimizer instance to schedule.
    Returns:
        scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler.
    """

    formatted_scheduler_name = cfg.scheduler.name.lower()

    if formatted_scheduler_name in ("none", "null"):
        print(f"Received `cfg.scheduler.name = {cfg.scheduler.name}`. Defaulting to NoSchedule.")
        scheduler_class = NoSchedule
    elif formatted_scheduler_name == "noschedule":
        scheduler_class = NoSchedule
    elif formatted_scheduler_name == "constant":
        scheduler_class = lr_scheduler.ConstantLR
    elif formatted_scheduler_name == "step":
        scheduler_class = lr_scheduler.StepLR
    elif formatted_scheduler_name == "multistep":
        scheduler_class = lr_scheduler.MultiStepLR
    elif formatted_scheduler_name == "linear":
        scheduler_class = lr_scheduler.LinearLR
    elif formatted_scheduler_name == "exponential":
        scheduler_class = lr_scheduler.ExponentialLR
    elif formatted_scheduler_name == "cosine":
        scheduler_class = lr_scheduler.CosineAnnealingLR
    elif formatted_scheduler_name == "cosine_restart":
        scheduler_class = lr_scheduler.CosineAnnealingWarmRestarts
    else:
        raise ValueError(f"Argument `cfg.scheduler.name` not recognized (got `{cfg.scheduler.name}`).")

    scheduler_obj = scheduler_class(optimizer, **cfg.scheduler.kwargs)

    return scheduler_obj