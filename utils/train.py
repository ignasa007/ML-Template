from typing import Dict

from torch import Tensor
from torch.optim import Optimizer

from models import BaseArchitecture
from metrics import Results


def train_batch(
    input,
    target: Tensor,
    model: BaseArchitecture,
    objective: str,
    optimizer: Optimizer,
    results: Results
) -> Dict[str, Tensor]:
    
    """
    Evaluate the model on a batch from the train set.
    Set train/eval mode before passing the model.

    Args:
        input: model input.
        target (torch.Tensor): groundtruth label.
        model (model.BaseModel): model.
        objective (str): key for the metrics dictionary returned by update step of `results`.
        optimizer (torch.optim.Optimizer): optimizer.
        results (Dict[str, torch.Tensor]): metrics over the batch.
    """

    out = model(input)
    metrics = results.forward(out, target)

    metrics[objective].backward()
    optimizer.step()

    return metrics