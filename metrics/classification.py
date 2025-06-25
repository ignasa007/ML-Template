from yacs.config import CfgNode
from torch import Tensor, sigmoid, softmax
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torchmetrics

from metrics.base import BaseMetric


class CELoss(BaseMetric):

    name = 'Cross Entropy Loss'

    def __init__(self, cfg: CfgNode):

        num_classes = cfg.dataset.num_classes
        if not isinstance(num_classes, int):
            raise TypeError(f'Expected `cfg.dataset.num_classes` to be an instance of `int` (got {type(num_classes)}).')

        if num_classes == 2:
            self.loss_fn = lambda out, target: binary_cross_entropy_with_logits(out, target.float(), reduction='sum')
        elif num_classes > 2:
            self.loss_fn = lambda out, target: cross_entropy(out, target, reduction='sum')
        else:
            raise ValueError(f'Expected `cfg.dataset.num_classes` to be >1 (got {num_classes}).')

        super(CELoss, self).__init__()

    def reset(self) -> None:
        self.total_loss = 0
        self.total_samples = 0

    def forward(self, out: Tensor, target: Tensor) -> Tensor:

        # Squeeze to make the output compatible for BCE loss
        out = out.squeeze()

        # Loss expects logits, not probabilities
        batch_total_loss = self.loss_fn(out, target)
        self.total_loss += batch_total_loss
        n_samples = target.size(0)
        self.total_samples += n_samples

        # Return average loss
        batch_avg_loss = batch_total_loss / n_samples

        return batch_avg_loss

    def compute(self) -> Tensor:
        avg_loss = self.total_loss / self.total_samples
        return avg_loss


class ClassificationMetric(BaseMetric):

    name: str
    underlying_metric: torchmetrics.Metric

    def __init__(self, underlying_metric_class, cfg):

        num_classes = cfg.dataset.num_classes
        if not isinstance(num_classes, int):
            raise TypeError(
                f"Expected `num_classes` to be an `int` greater than 1" \
                f" (got {num_classes=} of type {type(num_classes).__name__})."
            )

        if num_classes == 2:
            self.compute_proba = sigmoid
            self.underlying_metric = underlying_metric_class(task="binary")
        else:
            self.compute_proba = lambda probs: softmax(probs, dim=-1)
            self.underlying_metric = underlying_metric_class(task="multiclass", num_classes=num_classes)

        super(ClassificationMetric, self).__init__()

    def reset(self) -> None:
        self.underlying_metric.reset()

    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        # Other metrics expect (rather, can work with) probabilities, but not logits
        proba = self.compute_proba(out)
        metric = self.underlying_metric.forward(proba, target)
        return metric

    def compute(self) -> Tensor:
        return self.underlying_metric.compute()


class Accuracy(ClassificationMetric):
    name = 'Accuracy'
    def __init__(self, cfg: CfgNode):
        super(Accuracy, self).__init__(torchmetrics.Accuracy, cfg)

class F1Score(ClassificationMetric):
    name = 'F1 Score'
    def __init__(self, cfg: CfgNode):
        super(Accuracy, self).__init__(torchmetrics.F1Score, cfg)

class AUROC(ClassificationMetric):
    name = 'AUROC'
    def __init__(self, cfg: CfgNode):
        super(Accuracy, self).__init__(torchmetrics.AUROC, cfg)