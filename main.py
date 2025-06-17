import argparse
import os
from tqdm import tqdm

import torch

from data import get_datasets, get_loaders
from metrics import Results
from models import get_architecture
import sys; sys.exit()
from optimizers import get_optimizer
from utils import Config, Logger
from utils.train import train_batch
from utils.eval import eval_epoch


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="dataset name")
parser.add_argument("--architecture", type=str, required=True, help="architecture name")
parser.add_argument("--device_index", type=int, help="GPU device index")
parser.add_argument(
    "opts", default=None, nargs=argparse.REMAINDER, 
    help="modify config options using the command-line"
)
args = parser.parse_args()
cfg = Config(root="configs", args=args)
logger = Logger(args, cfg)

device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() and args.device_index is not None else "cpu")

logger.log("Loading and pre-processing datasets...", print_text=True)
# Datasets that are small enough to fit on the GPU should transfer the relevant objects using the `.to` method,
#   ... while others should have a no-op `.to` method
train_dataset, val_dataset, test_dataset = map(lambda dataset: dataset.to(device), get_datasets(args.dataset, cfg))
logger.log("Finished pre-processing datasets.\n", print_text=True)

logger.log("Preparing data-loaders...", print_text=True)
# If the dataset has already been pushed to the GPU, don't need to `pin_memory` in the loader
#   ... but if the dataset is too big to fit on the GPU, put the collated samples at pinned memory for GPU to pick from 
train_loader, val_loader, test_loader = get_loaders(train_dataset, val_dataset, test_dataset, cfg, device)
logger.log("Finished preparing data-loaders.\n", print_text=True)

logger.log("Preparing model...", print_text=True)
model = get_architecture(args.architecture, cfg).to(device)
logger.log("Finshed preparing model.\n", print_text=True)

optimizer = get_optimizer(cfg, model.parameters())
train_results, eval_results = Results(cfg).to(cfg.device), Results(cfg).to(cfg.device)
n_batches = 0
total_batches = len(train_loader) * cfg.dataset.n_epochs

# Ensure you log eval metrics only if train metrics have been logged (for convenience)
assert cfg.eval.log_every.batch % cfg.train.log_every.batch == 0
assert cfg.eval.log_every.epoch % cfg.train.log_every.epoch == 0

# Start training
for n_epochs in range(1, cfg.dataset.n_epochs+1):
    
    train_results.reset()
    
    # Train epoch
    for input, target in tqdm(train_loader):
        
        # Train batch
        metrics = train_batch(input, target, model, cfg.data.objective, optimizer, train_results)
        n_batches += 1

        # Log trained batch
        if cfg.train.log_every.batch > 0 and n_batches % cfg.train.log_every.batch == 0:
            logger.log(f"{n_batches} batches trained.")
            logger.log_metrics(metrics, prefix="Training: ".ljust(12))
            # Evaluate...
            if cfg.eval.log_every.batch > 0 and n_batches % cfg.eval.log_every.batch == 0:
                # ... on validation set...
                eval_results.reset()
                metrics = eval_epoch(val_loader, model, eval_results)
                logger.log_metrics(metrics, prefix="Validation: ".ljust(12))
                # ... and testing set
                eval_results.reset()
                metrics = eval_epoch(test_loader, model, eval_results)
                logger.log_metrics(metrics, prefix="Testing: ".ljust(12))
            # Print an empty line
            logger.log("", with_time=False)

        # Save model checkpoint
        if cfg.save_every.batch > 0 and n_batches % cfg.save_every.batch == 0:
            ckpt_fn = f"ckpt_batch-{n_batches}.pth"
            logger.save_tensors(kwargs={os.path.join(logger.EXP_DIR, ckpt_fn): model.state_dict()})
            logger.log(f"Saved checkpoint at {ckpt_fn}\n")
    
    # Log trained epoch
    if cfg.train.log_every.epoch > 0 and n_epochs % cfg.train.log_every.epoch == 0:
        logger.log(f"{n_epochs} epochs trained.")
        # Aggregate metrics over the entire epoch
        metrics = train_results.compute()
        logger.log_metrics(metrics, prefix="Training: ".ljust(12))
        # Evaluate...
        if cfg.eval.log_every.epoch > 0 and n_epochs % cfg.eval.log_every.epoch == 0:
            # ... on validation set...
            eval_results.reset()
            metrics = eval_epoch(val_loader, model, eval_results)
            logger.log_metrics(metrics, prefix="Validation: ".ljust(12))
            # ... and testing set
            eval_results.reset()
            metrics = eval_epoch(test_loader, model, eval_results)
            logger.log_metrics(metrics, prefix="Testing: ".ljust(12))
        # Print an empty line
        logger.log("", with_time=False)

    # Save model checkpoint
    if cfg.save_every.epoch > 0 and n_epochs % cfg.save_every.epoch == 0:
        ckpt_fn = f"ckpt_epoch-{n_epochs}.pth"
        logger.save_tensors(kwargs={os.path.join(logger.EXP_DIR, ckpt_fn): model.state_dict()})
        logger.log(f"Saved checkpoint at {ckpt_fn}\n")