# TODO: check if passing model to train functions makes copies. if so, refactor.


from typing import Callable

loss_fn: Callable


def train_batch(model, optimizer, *input_args, **input_kwargs):

    '''
    Evaluate the model on a batch from the train set.
    '''
    
    # set model into train mode
    model.train()
    optimizer.zero_grad()

    output = model(*input_args, **input_kwargs)
    loss = loss_fn(output)

    loss.backward()
    optimizer.step()

    return # for example, batch loss and/or predictions


def train_epoch(model, optimizer, data_loader):

    '''
    Evaluate the model on the training set.

    Args:
        data_loader (torch.utils.data.DataLoader): train loader.
    '''
    
    # initialize loggers, for example, total epoch loss and/or predictions
    
    for collated_batch in data_loader:
        # process the collated batch
        output = train_batch(model, optimizer, collated_batch)
        # update the loggers
        pass

    return # loggers