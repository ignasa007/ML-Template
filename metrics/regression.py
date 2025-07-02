from yacs.config import CfgNode
from torch import Tensor
import torchmetrics

from metrics.base import BaseMetric


class RegressionMetric(BaseMetric):
    # TODO: multi-dimensional regression?
    name: str
    underlying_metric: torchmetrics.Metric

    def __init__(self, underlying_metric_class, cfg):
        self.underlying_metric = underlying_metric_class()

    def reset(self) -> None:
        self.underlying_metric.reset()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        preds = preds.reshape(target.shape)
        value = self.underlying_metric.forward(preds, target)
        return value

    def compute(self) -> Tensor:
        return self.underlying_metric.compute()


class MeanSquaredError(RegressionMetric):
    name = "Mean Squared Error"
    def __init__(self, cfg: CfgNode):
        super(MeanSquaredError, self).__init__(torchmetrics.MeanSquaredError, cfg)

class MeanAbsoluteError(RegressionMetric):
    name = "Mean Absolute Error"
    def __init__(self, cfg: CfgNode):
        super(MeanAbsoluteError, self).__init__(torchmetrics.MeanAbsoluteError, cfg)

class MeanAbsolutePercentageError(RegressionMetric):
    name = "Mean Absolute Percentage Error"
    def __init__(self, cfg: CfgNode):
        super(MeanAbsolutePercentageError, self).__init__(torchmetrics.MeanAbsolutePercentageError, cfg)