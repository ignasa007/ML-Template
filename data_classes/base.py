class BaseDataset:

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        # should return a sample, for example, (image_i, label_i)
        raise NotImplementedError

    def __iter__(self):
        # should return an iterable object that can yield samples
        raise NotImplementedError