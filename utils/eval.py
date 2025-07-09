from typing import Dict

import torch
from torch.utils.data import DataLoader

from metrics import ResultsTracker


def eval_batch(
    input,
    target: torch.Tensor,
    model: torch.nn.Module,
    results: ResultsTracker
) -> Dict[str, torch.Tensor]:

    """
    Evaluate the model on a batch from the test set.
    Set train/eval mode before passing the model.

    Args:
        input: model input.
        target (torch.Tensor): groundtruth label.
        model (torch.nn.Module): model.
        results (Dict[str, torch.Tensor]): metrics over the batch.
    """

    with torch.no_grad():
        out = model(input)
        metrics = results.forward(out, target)

    return metrics


def eval_epoch(
    data_loader: DataLoader,
    model: torch.nn.Module,
    results: ResultsTracker
) -> Dict[str, torch.Tensor]:

    """
    Evaluate the model on the validation/testing set.
    Set train/eval mode before passing the model.

    Args:
        data_loader (torch.utils.data.DataLoader): val/test loader.
        model (model.BaseModel): model.
        results (Dict[str, torch.Tensor]): metrics over the entire epoch.
    """

    results.reset()
    for input, target in data_loader:
        _ = eval_batch(input, target, model, results)
    metrics = results.compute()

    return metrics