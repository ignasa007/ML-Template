from typing import Tuple, List

from yacs.config import CfgNode
from torch import Tensor, device as Device

# Import all metric classes in order to triger metric registration via decorators
from . import classification, regression
from .registry import get_metric


class ResultsTracker:

    def __init__(self, cfg: CfgNode):
        # Optimization objective
        self.metrics = [get_metric(cfg.dataset.objective, cfg)]
        # Other metrics to track
        for metric_name in cfg.dataset.metrics:
            self.metrics.append(get_metric(metric_name, cfg))

    def to(self, device: Device):
        for metric in self.metrics:
            metric = metric.to(device)
        return self

    def reset(self):
        for metric in self.metrics:
            metric.reset()
        return self

    def forward(self, preds: Tensor, target: Tensor) -> List[Tuple[str, Tensor]]:
        output = list()
        for metric in self.metrics:
            name = metric.__class__.name
            value = metric.forward(preds, target)
            output.append((name, value))
        return output

    def compute(self) -> List[Tuple[str, Tensor]]:
        output = list()
        for metric in self.metrics:
            name = metric.__class__.name
            value = metric.compute()
            output.append((name, value))
        return output