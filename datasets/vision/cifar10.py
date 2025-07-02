from yacs.config import CfgNode
from torch import Tensor
from torchvision.datasets import CIFAR10 as CIFAR10Torch

from datasets.vision.transforms import compose as compose_transforms
from datasets.vision.utils import get_split_indices


def load_cifar10(cfg: CfgNode):

    common_kwargs = dict(root=cfg.dataset.root, download=True)
    train_kwargs = dict(zip(("transform", "target_transform"), compose_transforms(cfg, train=True)), **common_kwargs)
    eval_kwargs = dict(zip(("transform", "target_transform"), compose_transforms(cfg, train=False)), **common_kwargs)
    eval_sets = list()

    def subset(dataset: CIFAR10Torch, indices: Tensor):
        dataset.data = dataset.data[indices]
        dataset.targets = [dataset.targets[i] for i in indices]
        return dataset

    # SPLIT TRAINING SET
    train_set = CIFAR10Torch(train=True, **train_kwargs)
    total_size = len(train_set)
    train_indices, [val_indices,] = get_split_indices(
        main_size=cfg.dataset.sizes.train,
        other_sizes=[cfg.dataset.sizes.val,],
        total_size=total_size,
    )
    train_set = subset(train_set, train_indices)

    # SPLIT VALIDATION SET
    if val_indices.nelement() > 0:
        val_set = CIFAR10Torch(train=True, **eval_kwargs)
        val_set = subset(val_set, val_indices)
        eval_sets.append(("Validation", val_set))

    # SPLIT TEST SET
    if cfg.dataset.sizes.test != 0.:
        test_set = CIFAR10Torch(train=False, **eval_kwargs)
        test_indices, _ = get_split_indices(
            main_size=cfg.dataset.sizes.test,
            other_sizes=None,
            total_size=len(test_set),
        )
        if test_indices.nelement() > 0:
            test_set = subset(test_set, test_indices)
            eval_sets.append(("Testing", test_set))

    # PIL format (H, W, C); layers expect (B, C, H, W) -- transform ToTensor handles it
    train_set.input_dim = train_set[0][0].size()

    return train_set, eval_sets