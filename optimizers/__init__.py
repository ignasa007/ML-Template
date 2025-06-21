from yacs.config import CfgNode
from torch.optim import Optimizer, SGD, RMSprop, Adam, AdamW


map = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adam': Adam,
    'adamw': AdamW,
}

def get_optimizer(parameters: dict, cfg: CfgNode) -> Optimizer:
    """
    Function to map optimizer name to optimizer class.
    Args:
        cfg (CfgNode): configurations picked from ./config/.
        parameters (dict): model parameters to optimize.
    Return:
        optimizer_class (optim.Optimizer): optimization algorithm.
    """

    formatted_optimizer_name = cfg.optimizer.name.lower()
    if formatted_optimizer_name not in map:
        raise ValueError(
            "Parameter `optimizer_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_optimizer_name},' for valid_optimizer_name in map.keys()) +
            f"\nbut got `{cfg.optimizer.name}`."
        )
    
    optimizer_class = map.get(formatted_optimizer_name)
    
    return optimizer_class(parameters, **dict(cfg.optimizer))