from typing import Any
from argparse import Namespace

from yacs.config import CfgNode


def default_cfg() -> CfgNode:

    """
    The default configuration object for the experiments.
        - Need to register all configurations that are expected.

    Returns:
        _C: A configuration object with placeholder values.
    """

    _C = CfgNode(new_allowed=True)
    _C.exp_dir = None

    # Dataset parameters
    _C.dataset = CfgNode()
    _C.dataset.root = None
    _C.dataset.output_dim = None
    # Dataset split sizes
    _C.dataset.sizes = CfgNode()
    _C.dataset.sizes.train = None
    _C.dataset.sizes.val = None
    _C.dataset.sizes.test = None
    # Transforms
    _C.dataset.transforms = CfgNode()
    _C.dataset.transforms.train = CfgNode()
    _C.dataset.transforms.train.input = None
    _C.dataset.transforms.train.target = None
    _C.dataset.transforms.eval = CfgNode()
    _C.dataset.transforms.eval.input = None
    _C.dataset.transforms.eval.target = None
    # Hardware-dependent configurations
    _C.dataset.batch_size = None
    _C.dataset.to_cuda = None
    # Metric tracking
    _C.dataset.objective = None
    _C.dataset.metrics = None
    # Dataloader configurations
    _C.loader = CfgNode()
    _C.loader.shuffle = None
    _C.loader.num_workers = None
    _C.loader.pin_memory = None

    # Architecture configurations
    _C.architecture = CfgNode()

    # Algorithm configurations
    _C.optimizer = CfgNode()
    _C.optimizer.name = None
    _C.scheduler = CfgNode()
    _C.scheduler.name = None
    _C.scheduler.step = CfgNode()
    _C.scheduler.step.batches = None
    _C.scheduler.step.epochs = None

    # Training configurations
    _C.train = CfgNode()
    _C.train.accum_grad = None
    # Stopping criterion
    _C.train.stop = CfgNode()
    _C.train.stop.batches = None
    _C.train.stop.epochs = None
    # Logging criterion
    _C.train.log = CfgNode()
    _C.train.log.batches = None
    _C.train.log.epochs = None
    
    # Evaluation configurations
    _C.eval = CfgNode()
    _C.eval.batches = None
    _C.eval.epochs = None
    # Model saving configurations
    _C.save = CfgNode()
    _C.save.batches = None
    _C.save.epochs = None

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