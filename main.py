import argparse
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_classes import dataclass_map
from model_classes import modelclass_map
from utils.config import Config
from utils.logger import Logger
from utils.train import train_batch
from utils.test import test_epoch 
from utils.metrics import Results


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


logger.log('Loading and pre-processing datasets...')
# load datasets
# preprocess datasets
logger.log('Finished pre-processing datasets.\n')


logger.log('Preparing data-loaders...')
# prepare data-loaders
train_loader: DataLoader
val_loader: DataLoader
test_loader: DataLoader
logger.log('Finished preparing data-loaders.\n')


logger.log('Loading and preparing model...')
# initiate model
model: torch.nn.Module
logger.log('Finshed preparing model.\n')

optimizer = Adam(model.parameters(), lr=cfg.LR)


logger.log(f"Starting training...\n", print_text=True)
# initialize results loggers
results = Results()
log_train = log_val = log_test = True
save_model = False

for epoch in range(cfg.DATA.N_EPOCHS):

    for collated_batch in tqdm(train_loader):

        train_batch(model, optimizer, collated_batch)
        results.update_results()

        if log_train:
            results.compute_metrics()
            # log metrics
        if log_val:
            test_epoch(model, val_loader)
            results.update_results()
            results.compute_metrics()
            # log metrics
        if log_test:
            test_epoch(model, test_loader, DEVICE)
            results.update_results()
            results.compute_metrics()
            # log metrics

        if save_model:
            ckpt_fn = f'{logger.SAVE_DIR}/ckpt.pth'
            logger.log(f'Saving model at {ckpt_fn}...', print_text=True)
            torch.save(model.state_dict(), ckpt_fn)
            logger.log(f'Finished saving model at {ckpt_fn}.\n', print_text=True)


for dataset in ('training', 'validation', 'testing'):
    logger.save(f'{dataset}_results', results.get(dataset))