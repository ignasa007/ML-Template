import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from data_classes import dataclass_map
from model_classes import modelclass_map
from utils.config import Config
from utils.logger import Logger
from utils.metrics import Results
from utils.test import test_epoch


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, 
    help='dataset directory name'
)
parser.add_argument(
    '--model', type=str, 
    help='hugging face model name'
)
parser.add_argument(
    '--weights', type=str, 
    help='path to learnt model weights'
)
parser.add_argument(
    'opts', default=None, nargs=argparse.REMAINDER, 
    help='modify config options using the command-line'
)
args = parser.parse_args()

cfg = Config(
    root='config', 
    dataset=args.dataset, 
    model=args.model,
    override=args.opts,
)

Dataset = dataclass_map(args.dataset)
Model = modelclass_map(args.model)
DEVICE = torch.device(f'cuda:{cfg.DEVICE_INDEX}' if torch.cuda.is_available() and cfg.DEVICE_INDEX is not None else 'cpu')

logger = Logger(
    dataset=args.dataset,
    model=cfg.MODEL.SAVE_NAME,
)

# log experiment configurations


logger.log('Loading and pre-processing dataset...')
# load datasets
# preprocess datasets
logger.log('Finished pre-processing dataset.\n')


logger.log('Preparing data-loader...')
# prepare data-loaders
test_loader: DataLoader
logger.log('Finished preparing data-loader.\n')


logger.log('Loading and preparing model...')
# initiate model
if os.path.exists(args.weights):
    if not args.weights.endswith('.pth'):
        logger.log(f'File {args.weights} is not a PyTorch state dictionary (must have a .pth extension).', print_text=True)
        sys.exit()
    else:
        # load trained weights
        model: torch.nn.Module
else:
    logger.log(f'File {args.weights} does not exist.', print_text=True)
    sys.exit()
logger.log('Finshed preparing model.\n')


results = Results()

logger.log('Starting inference...')
test_epoch(model, test_loader, DEVICE)
logger.log('Finished inference.')

results.update_results() # update results with the inference output
results.compute_metrics()
# log metrics