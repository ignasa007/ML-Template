from torch.utils.data import Dataset
from torch.utils.data import default_collate


class CustomDataset(Dataset):

    def __init__(self):

        super(CustomDataset, self).__init__()

    def __len__(self):
        
        raise NotImplementedError
    
    def __getitem__(self, index):
        
        raise NotImplementedError


def collate_fn(batch):

    '''
    Collate functions could be used for (amongst other things):
        1. applying a new random augmentations in each forward pass
        2. applying batch dependent transformations, eg. batch normalization
        3. simply preparing the input for the model

    Args:
        batch (List): a list of B samples.
            eg. [(image_1, label_1), ..., (image_B, label_B)]

    Return:
        loader_iterates (List): processed samples, as returned by data loader 
            eg. [(image_1, ..., image_B), (label_1, ..., label_B)]
    '''

    loader_iterates = default_collate(batch)

    return loader_iterates


def data_loaders(*args):
    
    '''
    Return data loaders for training, validation and test sets.
    '''
    
    raise NotImplementedError