import torch.nn as nn


def no_op(input):
    return input
    
class NoOp:
    def __init__(self, *args, **kwargs):
        ...
    def forward(self, input):
        return input

class NoOpModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOpModule, self).__init__()
    def forward(self, input):
        return input