from typing import Any, Tuple, List, Callable

from yacs.config import CfgNode
import torch.nn as nn


_ARCHITECTURE_REGISTRY = dict()

def register_architecture(*architecture_names: str) -> Callable:
    
    def decorator(cls):
        for architecture_name in architecture_names:
            if architecture_name in _ARCHITECTURE_REGISTRY:
                raise ValueError(f"`architecture_name={architecture_name}` already registered.")
            _ARCHITECTURE_REGISTRY[architecture_name] = cls
        return cls
    
    return decorator

def get_model(architecture_name: str, cfg: CfgNode) -> nn.Module:
    """
    Function to map architecture name to architecture class.
    Args:
        architecture_name (str): name of the architecture used for the experiment.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        architecture_objs (nn.Module): architecture object.
    """
    
    formatted_architecture_name = architecture_name.lower()
    if formatted_architecture_name not in _ARCHITECTURE_REGISTRY:
        raise ValueError(f"Argument `architecture_name` not recognized (got `{architecture_name}`).")
    
    architecture = _ARCHITECTURE_REGISTRY[formatted_architecture_name]
    model = architecture(cfg)
    
    return model

def list_architectures() -> List:
    
    return list(_ARCHITECTURE_REGISTRY.keys())