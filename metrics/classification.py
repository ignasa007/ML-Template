from typing import Union

from yacs.config import CfgNode
from torch import Tensor, sigmoid, softmax
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torchmetrics import Metric, Accuracy, F1Score, AUROC

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
    BaseMetric: Union[Accuracy, F1Score, AUROC]

    def __init__(self, cfg: CfgNode):

        num_classes = cfg.data.num_classes
        if not isinstance(self.num_classes, int):
            raise TypeError(f'Expected `cfg.dataset.num_classes` to be an instance of `int` (got {type(num_classes)}).')
        
        if num_classes == 2:
            self.compute_proba = sigmoid
            # Not sure if it is best practice to access BaseMetric through self
            self.base_metric = self.BaseMetric(task="binary")
        elif num_classes > 2:
            self.compute_proba = lambda probs: softmax(probs, dim=-1)
            self.base_metric = self.BaseMetric(task="multiclass", num_classes=num_classes)
        else:
            raise ValueError(f'Expected `cfg.dataset.num_classes` to be >1 (got {num_classes}).')
        
        super(ClassificationMetric, self).__init__()
        
    def reset(self) -> None:
        self.base_metric.reset()
    
    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        # Other metrics expect (rather, can work with) probabilities, but not logits
        proba = self.compute_proba(out)
        metric = self.base_metric.forward(proba, target)
        return metric

    def compute(self) -> Tensor:
        return self.base_metric.compute()
    

class Accuracy(ClassificationMetric):
    name = 'Accuracy'
    BaseMetric = Accuracy


class F1Score(ClassificationMetric):
    name = 'F1 Score'
    BaseMetric = F1Score
    

class AUROC(ClassificationMetric):
    name = 'AUROC'
    BaseMetric = AUROC