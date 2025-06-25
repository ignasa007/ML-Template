from typing import Optional, Tuple, List, Dict
from collections.abc import Iterable
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, get_loaders
from metrics import ResultsTracker
from models import BaseArchitecture, get_model
from algorithms import get_optimizer, get_scheduler
from utils import Config, Logger, eval_epoch


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
training_dataset, evaluation_datasets = get_dataset(args.dataset, cfg)
logger.log("Finished pre-processing datasets.\n", print_text=True)

logger.log("Preparing data-loaders...", print_text=True)
names, evaluation_datasets = zip(evaluation_datasets)
training_loader, evaluation_loaders = get_loaders([training_dataset]+list(evaluation_datasets), cfg, device)
evaluation_loaders = list(zip(names, evaluation_loaders))
logger.log("Finished preparing data-loaders.\n", print_text=True)

logger.log("Preparing model...", print_text=True)
model = get_model(args.architecture, cfg).to(device)
logger.log("Finshed preparing model.\n", print_text=True)

optimizer = get_optimizer(model.parameters(), cfg)
scheduler = get_scheduler(optimizer, cfg)
training_results_tracker, evaluation_results_tracker = ResultsTracker(cfg), ResultsTracker(cfg)
training_results_tracker.to(device), evaluation_results_tracker.to(device)

# HELPER FUNCTIONS

def log_metrics(
    logger: Logger,
    metrics: List[Tuple[str, Dict]],
    header: Optional[str] = None,
) -> None:

    if isinstance(header, str):
        logger.log(header)

    for set_name, set_metrics in metrics:
        logger.log_metrics(set_metrics, prefix=f"{set_name} set: ")

def evaluate_model(
    evaluation_loaders: Tuple[DataLoader],
    model: BaseArchitecture,
    evaluation_results_tracker: ResultsTracker,
    logger: Logger,
    header: Optional[str] = None,
) -> List[Tuple[str, List[Tuple[str, torch.Tensor]]]]:
    """
    Returns:
        metrics: [(set_name, [(metric_name, metric_value), ...]), ...]
        e.g. [
            (
                "Validation",
                [("Mean Squared Error", 0.01), ("Mean Absolute Error", 0.1)],
            ),
            (
                "Testing",
                [("Mean Squared Error", 0.01), ("Mean Absolute Error", 0.1)],
            ),
        ]
    """

    model.eval()
    metrics = list()

    for set_name, loader in evaluation_loaders:
        evaluation_results_tracker.reset()
        set_metrics = eval_epoch(loader, model, evaluation_results_tracker)
        metrics.append((set_name, set_metrics))

    log_metrics(logger, metrics, header=header)

    return metrics

def save_checkpoint(
    logger: Logger,
    state_dict: Dict,
    file_name: str,
    header: Optional[str] = None,
) -> None:

    if isinstance(header, str):
        logger.log(header)

    logger.save_tensors(kwargs={file_name: state_dict})
    logger.log(f"Saved checkpoint at {file_name}.")

###

n_epochs = n_batches = 0

# START TRAINING
while (
    (cfg.train.total_batches is None or n_batches < cfg.train.total_batches) and
    (cfg.train.total_epochs is None or n_epochs < cfg.train.total_epochs)
):

    # Book-keeping
    n_epochs += 1
    epoch_header = f"Epoch {n_epochs}"
    training_results_tracker.reset()

    for input, target in tqdm(training_loader):

        # Book-keeping
        n_batches += 1
        batch_header = f"Batch {n_batches}"
        evaluated_this_batch = False

        # FORWARD PROPAGATION
        model.train()
        output = model(input)
        # In case more than just the logits/regressands are outputted, e.g. activation maps
        if isinstance(output, Iterable):
            output = output[0]
        metrics = training_results_tracker.forward(output, target)

        # BACKWARD PROPAGATION
        objective_name, objective_value = metrics[0]
        objective_value.backward()
        if n_batches % cfg.train.update_every == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Gradient accumulation
            pass

        # UPDATE LEARNING RATE
        if isinstance(cfg.scheduler.step_every.batch, int) and (
            # Update learning rate every `cfg.scheduler.step_every.batch` batches trained
            cfg.scheduler.step_every.batch > 0 and n_batches % cfg.scheduler.step_every.batch == 0
        ):
            scheduler.step()

        # LOG BATCH METRICS
        if isinstance(cfg.train.log_every.batch, int) and (
            # Log batch metrics every `cfg.train.log_every.batch` batches trained
            cfg.train.log_every.batch > 0 and n_batches % cfg.train.log_every.batch == 0 or
            # Log batch metrics at the end of each epoch
            cfg.train.log_every.batch == -1 and hasattr(training_loader, "__len__") and n_batches % len(training_loader) == 0
        ):
            training_metrics = [("Training", metrics)]
            _ = log_metrics(logger, training_metrics, batch_header)
            batch_header = None

        # EVALUATE MODEL
        if isinstance(cfg.eval.log_every.batch, int) and (
            # Evaluate after every `cfg.eval.log_every.batch` batches trained
            cfg.eval.log_every.batch > 0 and n_batches % cfg.eval.log_every.batch == 0
        ):
            evaluation_metrics = evaluate_model(logger, model, evaluation_loaders, batch_header)
            batch_header, evaluated_this_batch = None, True

        # SAVE MODEL CHECKPOINT
        if isinstance(cfg.save_every.batch, int) and (
            # Save checkpoint after every `cfg.save_every.batch` batches trained
            cfg.save_every.batch > 0 and n_batches % cfg.save_every.batch == 0
        ):
            ckpt_file_name = f"ckpt_batch-{n_batches}.pth"
            _ = save_checkpoint(logger, model.state_dict(), ckpt_file_name, batch_header)
            batch_header = None

        # Empty line in case anything logged for this batch
        if batch_header is None:
            logger.log("", with_time=False)

    # UPDATE LEARNING RATE
    if isinstance(cfg.scheduler.step_every.epoch, int) and (
        # Update learning rate every `cfg.scheduler.step_every.epoch` epochs trained
        cfg.scheduler.step_every.epoch > 0 and n_epochs % cfg.scheduler.step_every.epoch == 0
    ):
        scheduler.step()

    # LOG EPOCH METRICS
    if isinstance(cfg.train.log_every.epoch, int) and (
        # Log aggregated training metrics every `cfg.train.log_every.epoch` epochs trained
        cfg.train.log_every.epoch > 0 and n_epochs % cfg.train.log_every.epoch == 0
    ):
        train_metrics = [("Training", training_results_tracker.compute())]
        _ = log_metrics(logger, train_metrics, epoch_header)
        epoch_header = None

    # EVALUATE MODEL
    if isinstance(cfg.eval.log_every.epoch, int) and (
        # Evaluate after every `cfg.eval.log_every.epoch` epochs trained
        cfg.eval.log_every.epoch > 0 and n_epochs % cfg.eval.log_every.epoch == 0 or
        # Evaluate at the end of training
        cfg.eval.log_every.epoch == -1 and n_epochs == cfg.train.total_epochs
    ):
        if evaluated_this_batch:
            log_metrics(logger, evaluation_metrics, epoch_header)
        else:
            _ = evaluate_model(logger, model, evaluation_loaders, epoch_header)
        epoch_header = None

    # SAVE MODEL CHECKPOINT
    if isinstance(cfg.save_every.epoch, int) and (
        # Save checkpoint after every `cfg.save_every.epoch` epochs trained
        cfg.save_every.epoch > 0 and n_epochs % cfg.save_every.epoch == 0 or
        # Save checkpoint at the end of training
        cfg.save_every.epoch == -1 and n_epochs == cfg.train.total_epochs
    ):
        ckpt_file_name = f"ckpt_epoch-{n_epochs}.pth"
        _ = save_checkpoint(logger, model.state_dict(), ckpt_file_name, epoch_header)
        epoch_header = None

    # Empty line in case anything logged for this epoch
    if epoch_header is None:
        logger.log("", with_time=False)