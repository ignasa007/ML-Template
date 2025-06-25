from typing import Tuple, List

from yacs.config import CfgNode
from torch import Tensor, device as Device

from metrics.base import BaseMetric
from metrics import classification, regression


map = {
    'celoss': classification.CELoss,
    'accuracy': classification.Accuracy,
    'f1score': classification.F1Score,
    'auroc': classification.AUROC,
    'mse': regression.MeanSquaredError,
    'mae': regression.MeanAbsoluteError,
    'mape': regression.MeanAbsolutePercentageError,
}


def get_metric(metric_name: str, cfg: CfgNode) -> BaseMetric:
    """
    Function to map metric name to metric class.
    Args:
        metric_name (str): name of the metric function.
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
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


class ResultsTracker:

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