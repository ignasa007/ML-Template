from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self):
        super(CustomDataset, self).__init__()
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass


def collate_fn(batch):
    # batch is a list of B samples
    #       for example, [(image_1, label_1), (image_2, label_2), ..., (image_B, label_B)]
    # collate functions could be (amongst other things) --
    #       1. run specific transformations, for example, random augmentations
    #       2. batch dependent transformations, for example, batch normalization
    #       3. simply preparing the input for the model
    #               for example, as ([image_1, ..., image_B], [label_1, ..., label_B])
    pass


def data_loaders(*args):
    # yield train_loader
    # yield val_loader
    # yield test_loader
    pass