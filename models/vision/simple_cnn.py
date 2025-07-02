from typing import Any, List, Optional

from yacs.config import CfgNode
import torch.nn as nn

from models.utils import get_activation
from models.vision.utils import get_normalization_layer, get_pooling_layer


# Copied from strtobool in distutils/util.py (deprecated Python 3.12 onwards)
def strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("Invalid truth value %r" % (val,))


def interpret_args(args: str, to: type, default: Any, num_layers: Optional[int] = None) -> List:
    """
    Parses args in format `type type*int type*int...`
    e.g. say `to` is `int`, then `args = "64 32*3 16*2"` is split as
        `out = [64, 32, 32, 32, 16, 16]`.
    e.g. say `to` is `bool`, then `args = "True*3 f*2 0"` is split as
        `out = [True, True, True, False, False False]`; see function `strtobool`.

    If args is split into one value only, then repeat it for `num_layers` number of times.
    e.g. `args=16, to=int, num_layers=3` is split as 
        `out = [16, 16, 16]`.
    """

    if to == bool:
        to = strtobool
    def cast(arg: str):
        if arg.lower() in ('none', 'null'):
            return None
        else:
            return to(arg)

    args = str(args)
    out = list()
    for arg in args.split():
        if not arg:
            continue
        elif "*" in arg:
            size, mult = arg.split("*")
            out.extend([cast(size)]*int(mult))
        else:
            out.append(cast(arg))
    
    if isinstance(num_layers, int) and num_layers < len(out):
        raise RuntimeError(f"Parsed more args than number of layers: {args = }, {num_layers=}, {out = }.")
    elif isinstance(num_layers, int) and num_layers > len(out):
        if len(out) == 1:
            out = out * num_layers
        else:
            raise RuntimeError(f"Cannot handle `1 < len(out) < num_layers`: {args = }, {num_layers=}, {out = }.")

    return out


def make_kwargs(**kwargs):
    for k, v in kwargs.items():
        if v is None:
            del kwargs[k]
    return kwargs


class SimpleCNN(nn.Module):
    # TODO: Add dropout

    def __init__(self, cfg: CfgNode):

        super(SimpleCNN, self).__init__()

        ### CONVOLUTIONAL BACKBONE
        
        num_channels = interpret_args(cfg.architecture.conv.num_channels, to=int, default=None, num_layers=None)
        num_conv_layers = len(num_channels)
        # Assuming square conv kernels
        kernel_sizes = interpret_args(cfg.architecture.conv.kernel_sizes, to=int, default=2, num_layers=num_conv_layers)
        strides = interpret_args(cfg.architecture.conv.strides, to=int, default=1, num_layers=num_conv_layers)
        paddings = interpret_args(cfg.architecture.conv.paddings, to=int, default=0, num_layers=num_conv_layers)
        biases = interpret_args(cfg.architecture.conv.biases, to=bool, default=True, num_layers=num_conv_layers)
        
        # CONV LAYERS
        self.conv_layers = nn.ModuleList()
        # Input layer; infer input size upon first forward pass
        kwargs = make_kwargs(kernel_size=kernel_sizes.pop(0), stride=strides.pop(0), padding=paddings.pop(0), bias=biases.pop(0))
        self.conv_layers.append(nn.LazyConv2d(num_channels[0], **kwargs))
        # Hidden layers
        for args in zip(num_channels[:-1], num_channels[1:], kernel_sizes, strides, paddings, biases):
            in_channels, out_channels, kernel_size, stride, padding, bias = args
            kwargs = make_kwargs(kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, **kwargs))

        # POOLING LAYERS
        pooler_names = interpret_args(cfg.architecture.conv.pooler.names, to=str, default=None, num_layers=num_conv_layers)
        # Assuming square pooling kernels
        kernel_sizes = interpret_args(cfg.architecture.conv.pooler.sizes, to=int, default=2, num_layers=num_conv_layers)
        strides = interpret_args(cfg.architecture.conv.pooler.strides, to=int, default=None, num_layers=num_conv_layers)
        paddings = interpret_args(cfg.architecture.conv.pooler.paddings, to=int, default=1, num_layers=num_conv_layers)
        self.conv_pooling_layers = nn.ModuleList()
        for args in zip(pooler_names, kernel_sizes, strides, paddings):
            pooler_name, kernel_size, stride, padding = args
            kwargs = make_kwargs(kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv_pooling_layers.append(get_pooling_layer(pooler_name, **kwargs))

        # NORMALIZATION LAYERS
        normalization_layers = interpret_args(cfg.architecture.conv.normalization_layers, to=str, default=None, num_layers=num_conv_layers)
        self.conv_normalization_layers = nn.ModuleList()
        for name in normalization_layers:
            self.conv_normalization_layers.append(get_normalization_layer(name))

        # ACTIVATION FUNCTIONS
        activations = interpret_args(cfg.architecture.conv.activations, to=str, default=None, num_layers=num_conv_layers)
        self.conv_activations = nn.ModuleList()
        for name in activations:
            self.conv_activations.append(get_activation(name))
        
        ### FULLY CONNECTED HEAD
        
        self.flatten = nn.Flatten()
        num_features = interpret_args(cfg.architecture.fcn.hidden_dims, to=int, default=None, num_layers=None)
        num_fcn_layers = len(num_features)
        biases = interpret_args(cfg.architecture.fcn.biases, to=bool, default=True, num_layers=num_fcn_layers+1)

        # FCN LAYERS
        self.fcn_layers = nn.ModuleList()
        # Input layer; infer input size upon first forward pass
        kwargs = make_kwargs(bias=biases.pop(0))
        self.fcn_layers.append(nn.LazyLinear(out_features=num_features[0], bias=biases.pop(0)))
        # Hidden layers
        for args in zip(num_features[:-1], num_features[1:], biases):
            in_features, out_features, bias = args
            kwargs = make_kwargs(bias=bias)
            self.fcn_layers.append(nn.Linear(in_features, out_features, **kwargs))
        # Output layer
        self.fcn_layers.append([
            nn.Linear(
                in_features=num_features[-1],
                out_features=cfg.dataset.output_dim,
                bias=biases.pop(0)
            )
        ])

        # ACTIVATION FUNCTIONS
        activations = interpret_args(cfg.architecture.fcn.activations, to=str, default=None, num_layers=num_fcn_layers)
        self.fcn_activations = nn.ModuleList()
        for name in activations:
            self.fcn_activations.append(get_activation(name))

    def reset_parameters(self):

        for layer in sum((
            self.conv_layers,
            self.conv_normalization_layers,
            self.fcn_layers
        )):
            layer.reset_parameters()

    def forward(self, x):

        for convolution, pooling, normalization, activation in zip(
            self.conv_layers, self.conv_pooling_layers, self.conv_normalization_layers, self.conv_activations
        ):
            # Order of ops is up to debate. My reasoning:
            #   1. Pooling commutes with monotonic activations => put pooling first to save compute.
            #   2. Normalization is put in diff places by diff authors.
            #       - Following the original paper, we perform norm before activation.
            #       - But after pooling to save compute, and retain its effect in the output.
            x = convolution(x)
            x = pooling(x)
            x = normalization(x)
            x = activation(x)

        x = self.flatten(x)
        for linear, activation in zip(self.fcn_layers, self.fcn_activations):
            x = linear(x)
            x = activation(x)

        return x