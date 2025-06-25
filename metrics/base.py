from torch import Tensor, device as Device


class BaseMetric:

    name: str

    def __init__(self):
        """Initialize an object to save the results in."""
        self.reset()

    def to(self, device: Device):
        """Compute the results on device."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        raise NotImplementedError

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Update the metric with batch predictions and target."""
        raise NotImplementedError

    def compute(self) -> Tensor:
        """Aggregate metric based on inputs passed in to `update_metric` previously."""
        raise NotImplementedError