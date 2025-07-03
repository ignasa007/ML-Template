from typing import Tuple, List

import torch
from torch.optim import Optimizer

from metrics import ResultsTracker


def train_batch(
    input,
    target: torch.Tensor,
    model: torch.nn.Module,
    optimizer: Optimizer,
    results: ResultsTracker
) -> List[Tuple[str, torch.Tensor]]:

    """
    Evaluate the model on a batch from the train set.
    Set train/eval mode before passing the model.

    Args:
        input: model input.
        target (torch.Tensor): groundtruth label.
        model (model.BaseModel): model being trained.
        optimizer (torch.optim.Optimizer): optimizer used for updating model parameters.
        results (List[Tuple[str, torch.Tensor]]): metrics over the batch.
    """

    out = model(input)
    metrics = results.forward(out, target)

    _, objective = metrics[0]
    objective.backward()
    optimizer.step()

    return metrics