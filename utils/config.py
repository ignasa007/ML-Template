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

    _C = CfgNode()
    _C.exp_dir = "./results"

    # Dataset parameters
    _C.dataset = CfgNode()
    _C.dataset.root = "./datastore"
    _C.dataset.output_dim = None
    # Transforms
    _C.dataset.transforms = CfgNode()
    _C.dataset.transforms.train = CfgNode()
    _C.dataset.transforms.train.input = list()
    _C.dataset.transforms.train.target = list()
    _C.dataset.transforms.eval = CfgNode()
    _C.dataset.transforms.eval.input = list()
    _C.dataset.transforms.eval.target = list()
    # Hardware-dependent configurations
    _C.dataset.batch_size = None
    _C.dataset.to_cuda = False
    # Metric tracking
    _C.dataset.objective = None
    _C.dataset.metrics = list()
    # Dataloader configurations
    _C.loader = CfgNode()
    _C.loader.shuffle = False
    _C.loader.num_workers = 1
    _C.loader.pin_memory = False

    # Architecture configurations
    _C.architecture = CfgNode()

    # Algorithm configurations
    _C.optimizer = CfgNode()
    _C.optimizer.name = "SGD"
    _C.optimizer.kwargs = CfgNode()
    _C.scheduler = CfgNode()
    _C.scheduler.name = "NoSchedule"
    _C.scheduler.kwargs = CfgNode()

    # Training configurations
    _C.training = CfgNode()
    _C.training.accum_grad = 1
    # Stopping criterion
    _C.training.stop = CfgNode()
    _C.training.stop.batches = None
    _C.training.stop.epochs = None
    # Logging criterion
    _C.training.log = CfgNode()
    _C.training.log.batches = None
    _C.training.log.epochs = None
    
    # Evaluation configurations
    _C.evaluation = CfgNode()
    _C.evaluation.batches = None
    _C.evaluation.epochs = None
    # Model saving configurations
    _C.save_ckpt = CfgNode()
    _C.save_ckpt.batches = None
    _C.save_ckpt.epochs = None

    return _C.clone()


def get_config(root: str, args: Namespace) -> CfgNode:

    """
    Args:
        root (str): file path for the default configurations.
        args (Namespace): command-line arguments.
    """

    cfg = default_cfg()
    cfg.set_new_allowed(True)

    cfg.merge_from_file(f"{root}/config.yaml")
    cfg.dataset.merge_from_file(f"{root}/datasets/{args.dataset.lower()}.yaml")
    cfg.architecture.merge_from_file(f"{root}/architectures/{args.architecture.lower()}.yaml")
    cfg.optimizer.merge_from_file(f"{root}/optimizers/{args.optimizer.lower()}.yaml")
    cfg.scheduler.merge_from_file(f"{root}/schedulers/{args.scheduler.lower()}.yaml")
    if isinstance(args.opts, list):
        cfg.merge_from_list(args.opts)

    return cfg