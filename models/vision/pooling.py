from typing import Dict

import torch.nn as nn

from ..common import NoOpModule


def get_pooling_layer(pooler_name: str, **kwargs: Dict) -> nn.Module:
    """
    Function to map pooler name to pooler object.
    Args:
        pooler_name (str): name of the pooler function.
        kwargs (Dict): pooler kwargs.
    Returns:
        pooler_obj (nn.Module): a pooler object.
    """

    formatted_pooler_name = str(pooler_name).lower()

    if formatted_pooler_name in ("none", "null"):
        pooler_class = NoOpModule
    elif formatted_pooler_name == "maxpool1d":
        pooler_class = nn.MaxPool1d
    elif formatted_pooler_name == "maxpool2d":
        pooler_class = nn.MaxPool2d
    elif formatted_pooler_name == "maxpool3d":
        pooler_class = nn.MaxPool3d
    elif formatted_pooler_name == "avgpool1d":
        pooler_class = nn.AvgPool1d
    elif formatted_pooler_name == "avgpool2d":
        pooler_class = nn.AvgPool2d
    elif formatted_pooler_name == "avgpool3d":
        pooler_class = nn.AvgPool3d
    else:
        raise ValueError(f"Argument `pooler_name` not recognized (got `{pooler_name}`).")

    pooler_obj = pooler_class(**kwargs)

    return pooler_obj