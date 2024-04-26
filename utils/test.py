# TODO: check if passing model to test functions makes copies. if so, refactor.

import torch


def test_batch(model, *input_args, **input_kwargs):
    
    # set model into eval mode
    model.eval()
    
    with torch.no_grad():
        output = model(*input_args, **input_kwargs)

    return # for example, batch loss and/or predictions


def test_epoch(model, data_loader):
    
    # initialize loggers, for example, total epoch loss and/or predictions
    
    for collated_batch in data_loader:
        # process the collated batch
        output = test_batch(model, collated_batch)
        # update the loggers
        pass

    return # loggers