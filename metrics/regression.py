from yacs.config import CfgNode
from torch import Tensor
from torchmetrics import Metric, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

from metrics.base import BaseMetric
    

class RegressionMetric(BaseMetric):
    # TODO: multi-dimensional regression?
    name: str
    BaseMetric: Metric
    
    def __init__(self, cfg: CfgNode):
        self.base_metric = self.BaseMetric()

    def reset(self) -> None:
        self.base_metric.reset()
    
    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        out = out.reshape(target.shape)
        metric = self.base_metric.forward(out, target)
        return metric
    
    def compute(self) -> Tensor:
        return self.base_metric.compute()


class MSE(RegressionMetric):
    name = 'Mean Squared Error'
    BaseMetric = MeanSquaredError
    

class MAE(RegressionMetric):
    name = 'Mean Absolute Error'
    BaseMetric = MeanAbsoluteError


class MAPE(RegressionMetric):
    name = 'Mean Absolute Percentage Error'
    BaseMetric = MeanAbsolutePercentageError


# class MAPE(MeanAbsolutePercentageError):

#     name = 'Mean Absolute Percentage Error'
    
#     def __init__(self, cfg: CfgNode):
#         super(MSE, self).__init__()
    
#     def forward(self, out: Tensor, target: Tensor):
#         out = out.reshape(target.shape)
#         metric = super(MAPE, self).forward(out, target)
#         return metric