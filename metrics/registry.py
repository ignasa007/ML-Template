from typing import List, Dict

from yacs.config import CfgNode

from .base import BaseMetric


_METRIC_REGISTRY: Dict[str, BaseMetric] = dict()

def register_metric(*metric_names: str):
    
    def decorator(cls):
        for metric_name in metric_names:
            if metric_name in _METRIC_REGISTRY:
                raise ValueError(f"`metric_name={metric_name}` already registered.")
            _METRIC_REGISTRY[metric_name] = cls
        return cls
    
    return decorator

def get_metric(metric_name: str, cfg: CfgNode) -> BaseMetric:
    """
    Function to map metric name to metric class.
    Args:
        metric_name (str): name of the metric.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        metric_class (BaseMetric): an evaluation metric.
    """
    
    formatted_metric_name = metric_name.lower()
    if formatted_metric_name not in _METRIC_REGISTRY:
        raise ValueError(f"Argument `metric_name` not recognized (got `{metric_name}`).")
    
    metric_class = _METRIC_REGISTRY[formatted_metric_name]
    metric_obj = metric_class(cfg)
    
    return metric_obj

def list_metrics() -> List:
    
    return list(_METRIC_REGISTRY.keys())