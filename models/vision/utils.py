"""
Implementing a per-task norm layer file because
    - I want lazy normalization layer implementations.
    - LazyBatchNorm automatically looks for input.size(1) from (N, C, H, W),
        since BatchNorm is used mainly for vision tasks.
    - Can't do the same with LazyLayerNorm because while vision tasks use (C, H, W) dims,
        NLP tasks usually use the embedding dim only.
"""

from typing import Dict

import torch.nn as nn


class NoOp(nn.Module):
    def forward(self, input):
        return input


def get_normalization_layer(normalization_layer_name: str, **kwargs: Dict) -> nn.Module:
    """
    Function to map normalization layer name to layer normalization object.
    Args:
        normalization_layer_name (str): name of the normalization layer.
    Returns:
        normalization_layer_obj (nn.Module): normalization layer object.
    """

    formatted_normalization_layer_name = str(normalization_layer_name).lower()

    if formatted_normalization_layer_name == "none":
        normalization_layer_class = NoOp
    elif formatted_normalization_layer_name == "batchnorm1d":
        normalization_layer_class = nn.LazyBatchNorm1d
    elif formatted_normalization_layer_name == "batchnorm2d":
        normalization_layer_class = nn.LazyBatchNorm2d
    elif formatted_normalization_layer_name == "batchnorm3d":
        normalization_layer_class = nn.LazyBatchNorm3d
    # TODO: implement lazy versions
    # elif formatted_normalization_layer_name == "layernorm":
    #     normalization_layer_class = nn.LazyLayerNorm
    # elif formatted_normalization_layer_name == "instancenorm":
    #     normalization_layer_class = nn.LazyInstanceNorm
    # elif formatted_normalization_layer_name == "groupnorm":
    #     normalization_layer_class = nn.LazyGroupNorm
    else:
        raise ValueError(f"Argument `normalization_layer_name` not recognized (got `{normalization_layer_name}`).")

    normalization_layer_obj = normalization_layer_class(**kwargs)

    return normalization_layer_obj


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

    if formatted_pooler_name == "none":
        print(f"Received `{pooler_name = }`. Defaulting to NoOp.")
        pooler_class = NoOp
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