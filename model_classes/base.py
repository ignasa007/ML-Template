import torch.nn as nn


class BaseModel(nn.Module):

    '''
    Base class for all model classes to import.
        - Inherits from nn.Module => can directly set model class
            in training/validation mode using model.train method.
        - The forward pass can be called using __call__.
    '''

    def __init__(self):

        '''
        Initialize the BaseModel class.
        '''

        super(BaseModel, self).__init__()

    def forward(self, *args):

        '''
        Forward propagate the inputs.
        '''

        raise NotImplementedError