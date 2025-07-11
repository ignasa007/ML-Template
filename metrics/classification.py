from yacs.config import CfgNode
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torchmetrics

from .base import BaseMetric
from .registry import register_metric


@register_metric("celoss")
class CELoss(BaseMetric):

    name = "Cross Entropy Loss"

    def __init__(self, cfg: CfgNode):

        output_dim = cfg.dataset.output_dim
        if not isinstance(output_dim, int):
            raise TypeError(f"Expected `cfg.dataset.output_dim` to be an `int` (got {type(output_dim)}).")

        if output_dim == 1:
            self.loss_fn = lambda out, target: binary_cross_entropy_with_logits(out, target.float(), reduction="sum")
        elif output_dim > 1:
            self.loss_fn = lambda out, target: cross_entropy(out, target, reduction="sum")
        else:
            raise ValueError(f"Expected `cfg.dataset.output_dim` to be >=1 (got {output_dim}).")

        super(CELoss, self).__init__()

    def to(self, device: torch.device) -> "CELoss":
        return self

    def reset(self) -> None:
        self.total_loss = 0
        self.total_samples = 0

    def forward(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # Squeeze to make the output compatible for BCE loss
        out = out.squeeze()

        # Squeeze to make the output compatible for BCE loss
        out = out.squeeze()
        # CE loss expects logits, not probabilities
        batch_loss = self.loss_fn(out, target)
        self.total_loss += batch_loss.detach()
        n_samples = target.size(0)
        self.total_samples += n_samples
        
        # Return average loss
        return batch_loss / n_samples

    def compute(self) -> torch.Tensor:
        # Return average loss
        return self.total_loss / self.total_samples


class ClassificationMetric(BaseMetric):

    name: str
    underlying_metric: torchmetrics.Metric

    def __init__(self, underlying_metric_class, cfg):

        output_dim = cfg.dataset.output_dim
        if not isinstance(output_dim, int):
            raise TypeError(f"Expected `num_classes` to be an `int` (got {type(output_dim).__name__}).")

        if output_dim == 1:
            self.compute_probs = torch.sigmoid
            self.underlying_metric = underlying_metric_class(task="binary")
        elif output_dim > 1:
            self.compute_probs = lambda probs: torch.softmax(probs, dim=-1)
            self.underlying_metric = underlying_metric_class(task="multiclass", num_classes=output_dim)
        else:
            raise ValueError(f"Expected `cfg.dataset.output_dim` to be >=1 (got {output_dim}).")
        super(ClassificationMetric, self).__init__()

    def to(self, device: torch.device) -> "ClassificationMetric":
        self.underlying_metric = self.underlying_metric.to(device)
        return self

    def reset(self) -> None:
        self.underlying_metric.reset()

    def forward(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Other metrics expect (rather, can work with) probabilities, but not logits
        probs = self.compute_probs(out)
        metric = self.underlying_metric.forward(probs, target)
        return metric

    def compute(self) -> torch.Tensor:
        return self.underlying_metric.compute()


@register_metric("accuracy")
class Accuracy(ClassificationMetric):
    name = "Accuracy"
    def __init__(self, cfg: CfgNode):
        super(Accuracy, self).__init__(torchmetrics.Accuracy, cfg)

@register_metric("f1score")
class F1Score(ClassificationMetric):
    name = "F1 Score"
    def __init__(self, cfg: CfgNode):
        super(F1Score, self).__init__(torchmetrics.F1Score, cfg)

@register_metric("auroc")
class AUROC(ClassificationMetric):
    name = "AUROC"
    def __init__(self, cfg: CfgNode):
        super(AUROC, self).__init__(torchmetrics.AUROC, cfg)