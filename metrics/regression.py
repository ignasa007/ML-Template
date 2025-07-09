from yacs.config import CfgNode
import torch
import torchmetrics

from .base import BaseMetric
from .registry import register_metric


class RegressionMetric(BaseMetric):
    
    name: str
    underlying_metric: torchmetrics.Metric

    def __init__(self, underlying_metric_class, cfg):
        self.underlying_metric = underlying_metric_class()

    def to(self, device: torch.device) -> "RegressionMetric":
        self.underlying_metric = self.underlying_metric.to(device)
        return self

    def reset(self) -> None:
        self.underlying_metric.reset()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds = preds.reshape(target.shape)
        value = self.underlying_metric.forward(preds, target)
        return value

    def compute(self) -> torch.Tensor:
        return self.underlying_metric.compute()


@register_metric("mse")
class MeanSquaredError(RegressionMetric):
    name = "Mean Squared Error"
    def __init__(self, cfg: CfgNode):
        super(MeanSquaredError, self).__init__(torchmetrics.MeanSquaredError, cfg)

@register_metric("mae")
class MeanAbsoluteError(RegressionMetric):
    name = "Mean Absolute Error"
    def __init__(self, cfg: CfgNode):
        super(MeanAbsoluteError, self).__init__(torchmetrics.MeanAbsoluteError, cfg)

@register_metric("mape")
class MeanAbsolutePercentageError(RegressionMetric):
    name = "Mean Absolute Percentage Error"
    def __init__(self, cfg: CfgNode):
        super(MeanAbsolutePercentageError, self).__init__(torchmetrics.MeanAbsolutePercentageError, cfg)