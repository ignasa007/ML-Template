"""
__init__ file for architecture classes.
Imports all the architecture classes and creates a function to map the 
    architecture name to the architecture class.
"""

from yacs.config import CfgNode

from models.architectures.base import BaseArchitecture


map = {
    # architecture name: architecture class
}


def get_architecture(architecture_name: str, cfg: CfgNode) -> BaseArchitecture:
    """
    Function to map architecture name to architecture class.
    Args:
        architecture_name (str): name of the architecture used for the experiment
    Return:
        architecture_class (Basearchitecture): a architecture class
    """
    
    formatted_architecture_name = architecture_name.lower()
    if formatted_architecture_name not in map:
        raise ValueError(
            "Parameter `architecture_name` not recognized. Expected one of" +
            "".join(f'\n\t- {valid_architecture_name},' for valid_architecture_name in map.keys()) +
            f"\nbut got `{architecture_name}`."
        )
    
    architecture_class = map.get(formatted_architecture_name)
    
    return architecture_class(cfg)