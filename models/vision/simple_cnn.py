from yacs.config import CfgNode
import torch
import torch.nn as nn

from ..registry import register_architecture
from ..common import get_activation, interpret_args, make_kwargs
from .pooling import  get_pooling_layer
from .normalization import get_normalization_layer


@register_architecture("simplecnn")
class SimpleCNN(nn.Module):

    def __init__(self, cfg: CfgNode):

        super(SimpleCNN, self).__init__()
        self.device = torch.device("cpu")

        ### CONVOLUTIONAL BACKBONE
        
        num_channels = interpret_args(cfg.architecture.conv.num_channels, to=int, num_layers=None)
        num_conv_layers = len(num_channels)
        # Assuming square conv kernels
        kernel_sizes = interpret_args(cfg.architecture.conv.kernel_sizes, to=int, num_layers=num_conv_layers)
        strides = interpret_args(cfg.architecture.conv.strides, to=int, num_layers=num_conv_layers)
        paddings = interpret_args(cfg.architecture.conv.paddings, to=int, num_layers=num_conv_layers)
        biases = interpret_args(cfg.architecture.conv.biases, to=bool, num_layers=num_conv_layers)
        
        # CONV LAYERS
        self.conv_layers = nn.ModuleList()
        if num_conv_layers > 0:
            # First layer; infer input size upon first forward pass
            kwargs = make_kwargs(kernel_size=kernel_sizes.pop(0), stride=strides.pop(0), padding=paddings.pop(0), bias=biases.pop(0))
            self.conv_layers.append(nn.LazyConv2d(num_channels[0], **kwargs))
        # Following layers
        for args in zip(num_channels[:-1], num_channels[1:], kernel_sizes, strides, paddings, biases):
            in_channels, out_channels, kernel_size, stride, padding, bias = args
            kwargs = make_kwargs(kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, **kwargs))

        # POOLING LAYERS
        pooler_names = interpret_args(cfg.architecture.conv.poolers.names, to=str, num_layers=num_conv_layers)
        # Assuming square kernels
        kernel_sizes = interpret_args(cfg.architecture.conv.poolers.sizes, to=int, num_layers=num_conv_layers)
        strides = interpret_args(cfg.architecture.conv.poolers.strides, to=int, num_layers=num_conv_layers)
        paddings = interpret_args(cfg.architecture.conv.poolers.paddings, to=int, num_layers=num_conv_layers)
        self.conv_pooling_layers = nn.ModuleList()
        for args in zip(pooler_names, kernel_sizes, strides, paddings):
            pooler_name, kernel_size, stride, padding = args
            kwargs = make_kwargs(kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv_pooling_layers.append(get_pooling_layer(pooler_name, **kwargs))

        # NORMALIZATION LAYERS
        normalization_layers = interpret_args(cfg.architecture.conv.normalization_layers, to=str, num_layers=num_conv_layers)
        self.conv_normalization_layers = nn.ModuleList()
        for name in normalization_layers:
            self.conv_normalization_layers.append(get_normalization_layer(name))

        # ACTIVATION FUNCTIONS
        activations = interpret_args(cfg.architecture.conv.activations, to=str, num_layers=num_conv_layers)
        self.conv_activations = nn.ModuleList()
        for name in activations:
            self.conv_activations.append(get_activation(name))
        
        ### FULLY CONNECTED HEAD
        
        self.flatten = nn.Flatten()
        num_features = interpret_args(cfg.architecture.fcn.hidden_dims, to=int, num_layers=None)
        num_fcn_layers = len(num_features)
        # activations will be the same length as the number of hidden layers
        activations = interpret_args(cfg.architecture.fcn.activations, to=str, num_layers=num_fcn_layers)
        if hasattr(cfg.dataset, "output_dim"):
            num_features += [cfg.dataset.output_dim]
            num_fcn_layers += 1
            activations.append("Identity")
        # biases will be the same length as the total number of layers (including output)
        # NOTE: bias in last layer unnecessary for CE loss, but can be relevant when using MSE
        biases = interpret_args(cfg.architecture.fcn.biases, to=bool, num_layers=num_fcn_layers)

        # FCN LAYERS
        self.fcn_layers = nn.ModuleList()
        if num_fcn_layers > 0:
            # First layer; infer input size upon first forward pass
            kwargs = make_kwargs(bias=biases.pop(0))
            self.fcn_layers.append(nn.LazyLinear(out_features=num_features[0], **kwargs))
        # Following layers
        for args in zip(num_features[:-1], num_features[1:], biases):
            in_features, out_features, bias = args
            kwargs = make_kwargs(bias=bias)
            self.fcn_layers.append(nn.Linear(in_features, out_features, **kwargs))

        # ACTIVATION FUNCTIONS
        self.fcn_activations = nn.ModuleList()
        for name in activations:
            self.fcn_activations.append(get_activation(name))

    def reset_parameters(self):

        for module in sum((
            self.conv_layers,
            self.conv_normalization_layers,
            self.fcn_layers
        )):
            module.reset_parameters()

    def to(self, device):

        self.device = device
        return super(SimpleCNN, self).to(device)

    def forward(self, x):

        for convolution, pooling, normalization, activation in zip(
            self.conv_layers, self.conv_pooling_layers, self.conv_normalization_layers, self.conv_activations
        ):
            # Order of ops is up to debate.
            x = convolution(x)
            x = normalization(x)
            x = activation(x)
            x = pooling(x)

        x = self.flatten(x)
        for linear, activation in zip(self.fcn_layers, self.fcn_activations):
            x = linear(x)
            x = activation(x)

        return x