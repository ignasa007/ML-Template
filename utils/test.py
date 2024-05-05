# TODO: check if passing model to test functions makes copies. if so, refactor.


import torch
from typing import Callable

loss_fn: Callable


def test_batch(model, *input_args, **input_kwargs):

    '''
    Evaluate the model on a batch from the test set.
    '''
    
    # set model into eval mode
    model.eval()
    
    with torch.no_grad():
        output = model(*input_args, **input_kwargs)
        loss = loss_fn(output)

    return # for example, batch loss and/or predictions


def test_epoch(model, data_loader):

    '''
    Evaluate the model on the validation/testing set.

    Args:
        data_loader (torch.utils.data.DataLoader): val/test loader.
    '''
    
    # initialize loggers, for example, total epoch loss and/or predictions
    
    for collated_batch in data_loader:
        # process the collated batch
        output = test_batch(model, collated_batch)
        # update the loggers
        pass

    return # loggers