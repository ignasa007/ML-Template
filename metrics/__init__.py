from typing import Dict

from yacs.config import CfgNode
from torch import Tensor, device as Device

from metrics.base import BaseMetric
from metrics.classification import *
from metrics.regression import *


map = {
    'celoss': CELoss,
    'accuracy': Accuracy,
    'f1score': F1Score,
    'auroc': AUROC,
    'mse': MSE,
    'mae': MAE,
    'mape': MAPE,
}


def get_metric(metric_name: str, cfg: CfgNode) -> BaseMetric:
    """
    Function to map metric name to metric class.
    Args:
        metric_name (str): name of the metric function.
        cfg (yacs.CfgNode): experiment configurations.
    Return:
        metric_class (BaseMetric): a piecewise metric function.
    """

    formatted_metric_name = metric_name.lower()
    if formatted_metric_name not in map:
        raise ValueError(
            "Parameter `metric_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_metric_name},' for valid_metric_name in map.keys()) +
            f"\nbut got `{metric_name}`."
        )
    
    metric_class = map.get(formatted_metric_name)
    
    return metric_class(cfg)


class Results:

    def __init__(self, cfg: CfgNode):
        # Optimization objective
        self.metrics = [get_metric(cfg.dataset.objective, cfg)]
        # Other metrics to track
        for metric_name in cfg.dataset.metrics:
            self.metrics.append(get_metric(metric_name, cfg))

    def to(self, device: Device) -> None:
        for metric in self.metrics:
            metric = metric.to(device)

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
    
    def forward(self, out: Tensor, target: Tensor) -> Dict[str, Tensor]:
        out = dict()
        for metric in self.metrics:
            out[metric.__class__.name] = metric.forward(out, target)
        return out
    
    def compute(self) -> Dict[str, Tensor]:
        out = dict()
        for metric in self.metrics:
            out[metric.__class__.name] = metric.compute()
        return out