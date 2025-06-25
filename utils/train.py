from typing import Tuple, List

from torch import Tensor
from torch.optim import Optimizer

from models import BaseArchitecture
from metrics import ResultsTracker


def train_batch(
    input,
    target: Tensor,
    model: BaseArchitecture,
    optimizer: Optimizer,
    results: ResultsTracker
) -> List[Tuple[str, Tensor]]:

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

    objective_name, objective_value = metrics[0]
    objective_value.backward()
    optimizer.step()

    return metrics