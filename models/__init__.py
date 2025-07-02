"""
__init__ file for architecture classes.
Imports all the architecture classes and creates a function to map the
    architecture name to the architecture class.
"""

from yacs.config import CfgNode

from models.base import BaseArchitecture
from models.vision import map as vision_map
from models.text import map as text_map
from models.graph import map as graph_map


map = {**vision_map, **text_map, **graph_map}

def get_model(architecture_name: str, cfg: CfgNode) -> BaseArchitecture:
    """
    Function to map architecture name to architecture class.
    Args:
        architecture_name (str): name of the architecture used for the experiment
        cfg (yacs.CfgNode): experiment configurations.
    Returns:
        architecture_obj (BaseArchitecture): an architecture object
    """

    formatted_architecture_name = architecture_name.lower()
    if formatted_architecture_name not in map:
        raise ValueError(
            "Parameter `architecture_name` not recognized. Expected one of" +
            "".join(f'\n\t- {name},' for name in map.keys()) +
            f"\nbut got `{architecture_name}`."
        )

    architecture_class = map.get(formatted_architecture_name)
    architecture_obj = architecture_class(cfg)

    return architecture_obj