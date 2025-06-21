import argparse
import os
from tqdm import tqdm

import torch

from datasets import get_dataset, get_loaders
from metrics import Results
from models import get_architecture
import sys; sys.exit()
from optimizers import get_optimizer
from utils import Config, Logger
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
train_dataset, eval_datasets = get_dataset(args.dataset, cfg)
logger.log("Finished pre-processing datasets.\n", print_text=True)

logger.log("Preparing data-loaders...", print_text=True)
names, eval_datasets = zip(eval_datasets)
train_loader, eval_loaders = get_loaders([train_dataset]+list(eval_datasets), cfg, device)
eval_loaders = [(name, eval_loader) for name, eval_loader in zip(names, eval_loaders)]
logger.log("Finished preparing data-loaders.\n", print_text=True)

logger.log("Preparing model...", print_text=True)
model = get_architecture(args.architecture, cfg).to(device)
logger.log("Finshed preparing model.\n", print_text=True)

optimizer = get_optimizer(model.parameters(), cfg)
train_results, eval_results = Results(cfg), Results(cfg)
train_results.to(device), eval_results.to(device)

# Ensure you log eval metrics only if train metrics have been logged (for convenience)
assert cfg.eval.log_every.batch % cfg.train.log_every.batch == 0
assert cfg.eval.log_every.epoch % cfg.train.log_every.epoch == 0
# Ensure you save model only if eval metrics have been logged
assert cfg.save_every.batch % cfg.eval.log_every.batch == 0
assert cfg.save_every.epoch % cfg.eval.log_every.epoch == 0

n_epochs = n_batches = 0

# START TRAINING
while n_epochs < cfg.train.n_epochs:
    
    # Train epoch
    n_epochs += 1
    train_results.reset()

    for input, target in tqdm(train_loader):
        
        # FORWARD PROPAGATION
        n_batches += 1
        out = model(input)
        metrics = train_results.forward(out, target)

        # BACKWARD PROPAGATION
        metrics[cfg.dataset.objective].backward()
        if n_batches % cfg.train.update_every == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Gradient accumulation
            pass

        # LOGGING AND EVALUATION
        if cfg.train.log_every.batch > 0 and n_batches % cfg.train.log_every.batch == 0:
            # Log trained batch
            logger.log(f"{n_batches} batches trained.")
            logger.log_metrics(metrics, prefix="Training set: ")
            if 0 < cfg.eval.log_every.batch and n_batches % cfg.eval.log_every.batch == 0:
                # Evaluate and log
                for i, (name, eval_loader) in enumerate(eval_loaders, 1):
                    eval_results.reset()
                    metrics = eval_epoch(eval_loader, model, eval_results)
                    logger.log_metrics(metrics, prefix=f"{name} set: ")
                # Save model checkpoint
                if cfg.save_every.batch > 0 and n_batches % cfg.save_every.batch == 0:
                    ckpt_fn = f"ckpt_batch-{n_batches}.pth"
                    logger.save_tensors(kwargs={ckpt_fn: model.state_dict()})
                    logger.log(f"Saved checkpoint {ckpt_fn}.")
            logger.log("", with_time=False)

        # If total training batches specified, check if training should be stopped
        if n_batches >= cfg.train.n_batches:
            break
    
    # LOGGING AND EVALUATION
    if cfg.train.log_every.epoch > 0 and n_epochs % cfg.train.log_every.epoch == 0:
        # Log trained epoch
        logger.log(f"{n_epochs} epochs trained.")
        # Aggregate metrics over the entire epoch
        metrics = train_results.compute()
        logger.log_metrics(metrics, prefix="Training Set: ")
        if cfg.eval.log_every.epoch > 0 and n_epochs % cfg.eval.log_every.epoch == 0:
            # Evaluate and log
            for i, eval_loader in enumerate(eval_loaders, 1):
                eval_results.reset()
                metrics = eval_epoch(eval_loader, model, eval_results)
                logger.log_metrics(metrics, prefix=f"Evaluation Set {i:0{len(eval_loaders)}d}: ")
            # Save model checkpoint
            if cfg.save_every.epoch > 0 and n_epochs % cfg.save_every.epoch == 0:
                ckpt_fn = f"ckpt_epoch-{n_epochs}.pth"
                logger.save_tensors(kwargs={ckpt_fn: model.state_dict()})
                logger.log(f"Saved checkpoint {ckpt_fn}.")
        logger.log("", with_time=False)