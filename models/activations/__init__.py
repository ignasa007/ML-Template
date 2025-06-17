import torch.nn as nn


map = {
    'identity': nn.Identity,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
}

def get_activation(activation_name: str) -> nn.Module:
    """
    Function to map activation name to activation class.
    Args:
        activation_name (str): name of the activation function
    Return:
        activation_class (nn.Module): a piecewise activation function
    """

    formatted_activation_name = activation_name.lower()
    if formatted_activation_name not in map:
        raise ValueError(
            "Parameter `activation_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_activation_name},' for valid_activation_name in map.keys()) +
            f"\nbut got `{activation_name}`."
        )
    
    activation_class = map.get(formatted_activation_name)
    
    return activation_class