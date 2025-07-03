import torch.nn as nn


def get_activation(activation_name: str) -> nn.Module:
    """
    Function to map activation name to activation object.
    Args:
        activation_name (str): name of the activation function.
    Returns:
        activation_obj (nn.Module): a piecewise activation function.
    """

    formatted_activation_name = str(activation_name).lower()

    if formatted_activation_name in ("none", "null"):
        print(f"Received `{activation_name = }`. Defaulting to Identity.")
        activation_class = nn.Identity 
    elif formatted_activation_name == "identity":
        activation_class = nn.Identity
    elif formatted_activation_name == "relu":
        activation_class = nn.ReLU
    elif formatted_activation_name == "gelu":
        activation_class = nn.GELU
    elif formatted_activation_name == "elu":
        activation_class = nn.ELU
    elif formatted_activation_name == "sigmoid":
        activation_class = nn.Sigmoid
    elif formatted_activation_name == "tanh":
        activation_class = nn.Tanh
    else:
        raise ValueError(f"Argument `activation_name` not recognized (got `{activation_name}`).")
    
    activation_obj = activation_class()

    return activation_obj