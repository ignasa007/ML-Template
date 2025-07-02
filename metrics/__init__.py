from typing import Tuple, List

from yacs.config import CfgNode
from torch import Tensor, device as Device

from metrics.base import BaseMetric
from metrics import classification, regression


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

    if formatted_metric_name == "celoss":
        metric_class = classification.CELoss
    elif formatted_metric_name == "accuracy":
        metric_class = classification.Accuracy
    elif formatted_metric_name == "f1score":
        metric_class = classification.F1Score
    elif formatted_metric_name == "auroc":
        metric_class = classification.AUROC
    elif formatted_metric_name == "mse":
        metric_class = regression.MeanSquaredError
    elif formatted_metric_name == "mae":
        metric_class = regression.MeanAbsoluteError
    elif formatted_metric_name == "mape":
        metric_class = regression.MeanAbsolutePercentageError
    else:
        raise ValueError(f"Argument `metric_name` not recognized (got `{metric_name}`).")

    # TODO: Seems like an overkill to pass the entire cfg to `metric_class`; just need `num_classes` I think
    metric_obj = metric_class(cfg)

    return metric_obj


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