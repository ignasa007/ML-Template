# Import all dataset classes in order to triger dataset registration via decorators
from . import vision, text, graph
from .registry import BaseDataset, get_dataset

from .loaders import get_loaders