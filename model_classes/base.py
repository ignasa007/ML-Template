import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError