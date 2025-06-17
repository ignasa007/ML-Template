from typing import Any
from argparse import Namespace

from yacs.config import CfgNode


def default_cfg() -> CfgNode:

    """
    The default configuration object for the experiments.
        - Need to register all configurations that are expected.

    Return:
        _C: A configuration object with placeholder values.
    """

    _C = CfgNode()
    _C.device = None

    # Dataloader parameters
    _C.loader = CfgNode()
    _C.loader.shuffle = None
    _C.loader.num_workers = None
    _C.loader.pin_memory = None

    # Dataset parameters
    _C.dataset = CfgNode()
    _C.dataset.objective = None
    _C.dataset.metrics = None
    _C.dataset.batch_size = None
    _C.dataset.n_epochs = None

    # Architecture parameters
    _C.architecture = CfgNode()
    _C.architecture.width = None
    _C.architecture.depth = None
    _C.architecture.residual_conn = None

    return _C.clone()


class Config:
    
    def __init__(self, root: str, args: Namespace):

        """
        Initialization of the configuration object used by the main file.

        Args:
            root (str): file path for the default configurations.
            args (Namespace): command-line arguments.
        """

        self.cfg = default_cfg()

        self.cfg.merge_from_file(f"{root}/config.yaml")
        self.cfg.dataset.merge_from_file(f"{root}/datasets/{args.dataset}.yaml")
        self.cfg.architecture.merge_from_file(f"{root}/architectures/{args.architecture}.yaml")

        if isinstance(args.opts, list):
            self.cfg.merge_from_list(args.opts)

    def __getattr__(self, name: str) -> Any:
        """Method for returning configurations using dot operator."""
        return self.cfg.__getattr__(name)