from typing import Any, Tuple, Iterable, Callable, Union

from torch import Tensor, device as Device
from torch.utils.data import default_collate, Dataset


def base_preprocess(sample: Iterable[Any]):
    
    """
    Any methods/classes needed to preprocess the data.
    This is different from collate function in that collate processes a batch of samples at a time.
    Preprocessing is stuff like cleaning up text:
        1. it is usually performed upon loading the dataset, and is the same in each run (unlike, say, random augmentation)
        2. it does not require context from other samples (unlike, say, batch norm)

    Args:
        sample (Iterable): a collection of objects forming a sample.
            eg. (image_1, label_1)

    Return:
        dataset_iterate (Iterable): preprocessed sample
            eg. (preprocessed_image_1, preprocessed_label_1)
    """

    return sample


def base_collate(batch: Iterable[Iterable[Tensor]]):

    """
    Collate functions could be used for (amongst other things):
        1. applying a new random augmentations in each forward pass
        2. applying batch dependent transformations, eg. batch normalization
        3. simply preparing the input for the model

    Args:
        batch (Iterable): a list of B samples.
            eg. [(image_1, label_1), ..., (image_B, label_B)]

    Return:
        loader_iterates (Iterable): processed samples, as returned by data loader 
            eg. [(image_1, ..., image_B), (label_1, ..., label_B)]
    """

    loader_iterates = default_collate(batch)
    return loader_iterates


class BaseDataset(Dataset):
    """
    Base class for all data classes to import.
    
    Class attributes should be those that don't need to change from run to run,
        eg. preprocessor, collater

    In the corresponding config file, define hyperparameters that may be tuned,
        eg. batch size, subset size
    """

    preprocessor: Callable
    collater: Callable

    def __init__(self):
        """Initialize the BaseDataset class."""
        super(BaseDataset, self).__init__()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError
    
    def __getitem__(self, key: Union[int, str]) -> Tuple[Tensor, ...]:
        """Return a sample from the dataset, e.g. (image_i, label_i)."""
        raise NotImplementedError
    
    def to(self, device: Device) -> None:
        """Transfer the samples to `device`."""
        raise NotImplementedError