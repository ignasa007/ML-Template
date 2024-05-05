class BaseDataset:

    '''
    Base class for all data classes to import.
    '''

    def __init__(self):

        '''
        Initialize the BaseDataset class.
        '''

        raise NotImplementedError

    def __getitem__(self, key):
        
        '''
        Return a sample from the dataset, for example, 
            (image_i, label_i).
        '''

        raise NotImplementedError

    def __iter__(self):
        
        '''
        Return an iterable object that can yield samples.
        '''

        raise NotImplementedError