from typing import Tuple

from yacs.config import CfgNode
import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.datasets import CIFAR10 as CIFAR10Torch

from ..registry import register_dataset
from .transforms import compose as compose_transforms
from .utils import get_split_indices


class CIFAR10(CIFAR10Torch):

    storage_device = torch.device("cpu")
    target_device = torch.device("cpu")

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        # `self.targets[i]` still returns an `int`, just indexing is improved.
        self.targets = np.array(self.targets)

    def subset(self, indices: torch.Tensor):
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        return self

    def to(self, device: torch.device):
        # `self.data` and `self.targets` are NumPy arrays -> can't put either of them on GPU.
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements regular `__getitem__`, unless some transform has been defined which converts
            img/target to a `Tensor`, in which case the tensor is transferred to `target_device`.
        """
        img, target = super(CIFAR10, self).__getitem__(index)
        # Only transfer to `target_device`` if image/target retrieved as a Tensor.
        if hasattr(img, "to"):
            img = img.to(self.target_device)
        if hasattr(target, "to"):
            target = target.to(self.target_device)
        return img, target
    
    def collate(self, batch):
        """Implements `default_collate`, followed by transfer to `target_device`."""
        inputs, targets = default_collate(batch)
        inputs = inputs.to(self.target_device)
        targets = targets.to(self.target_device)
        return inputs, targets


@register_dataset("cifar10")
def load_cifar10(cfg: CfgNode):

    common_kwargs = dict(root=cfg.dataset.root, download=True)
    train_kwargs = dict(zip(("transform", "target_transform"), compose_transforms(cfg, train=True)), **common_kwargs)
    eval_kwargs = dict(zip(("transform", "target_transform"), compose_transforms(cfg, train=False)), **common_kwargs)
    eval_sets = list()

    # SPLIT TRAINING SET
    train_set = CIFAR10(train=True, **train_kwargs)
    total_size = len(train_set)
    train_indices, [val_indices,] = get_split_indices(
        main_size=cfg.dataset.sizes.train,
        other_sizes=[cfg.dataset.sizes.val,],
        total_size=total_size,
    )
    train_set = train_set.subset(train_indices)

    # SPLIT VALIDATION SET
    if val_indices.nelement() > 0:
        val_set = CIFAR10(train=True, **eval_kwargs)
        val_set = val_set.subset(val_indices)
        eval_sets.append(("Validation", val_set))

    # SPLIT TEST SET
    if cfg.dataset.sizes.test != 0.:
        test_set = CIFAR10(train=False, **eval_kwargs)
        test_indices, _ = get_split_indices(
            main_size=cfg.dataset.sizes.test,
            other_sizes=None,
            total_size=len(test_set),
        )
        if test_indices.nelement() > 0:
            test_set = test_set.subset(test_indices)
            eval_sets.append(("Testing", test_set))

    # PIL format (H, W, C); layers expect (B, C, H, W) -- transform ToTensor handles it
    input_dim = train_set[0][0].size()

    return input_dim, train_set, eval_sets