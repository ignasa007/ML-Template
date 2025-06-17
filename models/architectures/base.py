from yacs.config import CfgNode
from torch import Tensor
import torch.nn as nn


class BaseArchitecture(nn.Module):
    """Base class for all architecture classes to import."""

    def __init__(self, cfg: CfgNode):
        """Initialize the BaseArchitecture class."""
        super(BaseArchitecture, self).__init__()

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward propagate the inputs."""
        raise NotImplementedError