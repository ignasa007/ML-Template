from typing import Optional, Tuple, List, Dict
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, get_loaders
from metrics import ResultsTracker
from models import get_model
from algorithms import get_optimizer, get_scheduler
from utils import get_config, Logger, eval_epoch


# HELPER FUNCTIONS

def log_metrics(
    logger: Logger,
    metrics: List[Tuple[str, Dict]],
    header: Optional[str] = None,
) -> None:

    if isinstance(header, str):
        logger.log(header, with_time=False)

    for set_name, set_metrics in metrics:
        logger.log_metrics(set_metrics, prefix=f"{set_name} set: ")

def evaluate_model(
    evaluation_loaders: Tuple[DataLoader],
    model: torch.nn.Module,
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

# DRIVER CODE

def main(args):

    """
    The implementation of multiprocessing is different on Windows, which uses spawn instead of fork. 
    So we have to wrap the code with an if-clause to protect the code from executing multiple times.
        - https://docs.pytorch.org/docs/stable/notes/windows.html#usage-multiprocessing
    """

    cfg = get_config(root="configs", args=args)
    logger = Logger(args, cfg)
    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() and args.device_index is not None else "cpu")

    logger.log("Loading and pre-processing datasets...", print_text=True)
    input_dim, training_dataset, evaluation_datasets = get_dataset(args.dataset, cfg)
    # Not sure what the point is of this if we can use lazy initialization of modules
    cfg.dataset.input_dim = tuple(input_dim)
    logger.log("Finished pre-processing datasets.", print_text=True)

    logger.log("Preparing data-loaders...", print_text=True)
    names, evaluation_datasets = zip(*evaluation_datasets)
    training_loader, *evaluation_loaders = get_loaders(training_dataset, *evaluation_datasets, cfg=cfg, device=device)
    evaluation_loaders = list(zip(names, evaluation_loaders))
    logger.log("Finished preparing data-loaders.", print_text=True)

    logger.log("Preparing model...", print_text=True)
    model = get_model(args.architecture, cfg).to(device)
    logger.log("Finshed preparing model.\n", print_text=True)

    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)
    training_results_tracker, evaluation_results_tracker = ResultsTracker(cfg), ResultsTracker(cfg)
    training_results_tracker.to(device), evaluation_results_tracker.to(device)

    ###

    n_epochs = n_batches = 0

    # START TRAINING
    while (
        (cfg.training.stop.batches is None or n_batches < cfg.training.stop.batches) and
        (cfg.training.stop.epochs is None or n_epochs < cfg.training.stop.epochs)
    ):

        # Book-keeping
        n_epochs += 1
        epoch_header = f"Epoch {n_epochs}"
        training_results_tracker.reset()

        for inputs, targets in tqdm(training_loader):

            # Need to transfer to GPU here because I don't know multiprocessing :')
            # https://discuss.pytorch.org/t/dataset-location-runtimeerror-caught-runtimeerror-in-dataloader-worker-process-0/156842/4
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            # Book-keeping
            n_batches += 1
            batch_header = f"Batch {n_batches}"
            evaluated_this_batch = False

            # FORWARD PROPAGATION
            model.train()
            outputs = model(inputs)
            # In case more than just the logits/regressands are outputted, e.g. activation maps
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            metrics = training_results_tracker.forward(outputs, targets)

            # BACKWARD PROPAGATION
            objective_name, objective_value = metrics[0]
            objective_value.backward()
            if n_batches % cfg.training.accum_grad == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
            else:
                # Gradient accumulation
                pass

            # UPDATE LEARNING RATE
            if isinstance(cfg.scheduler.step.batches, int) and (
                # Update learning rate every `cfg.scheduler.step.batches` batches trained
                cfg.scheduler.step.batches > 0 and n_batches % cfg.scheduler.step.batches == 0
            ):
                scheduler.step()

            # LOG BATCH METRICS
            if isinstance(cfg.training.log.batches, int) and (
                # Log batch metrics every `cfg.training.log.batches` batches trained
                cfg.training.log.batches > 0 and n_batches % cfg.training.log.batches == 0 or
                # Log batch metrics at the end of each epoch
                cfg.training.log.batches == -1 and hasattr(training_loader, "__len__") and n_batches % len(training_loader) == 0
            ):
                training_metrics = [("Training", metrics)]
                _ = log_metrics(logger, training_metrics, batch_header)
                batch_header = None

            # EVALUATE MODEL
            if isinstance(cfg.evaluation.batches, int) and (
                # Evaluate after every `cfg.evaluation.batches` batches trained
                cfg.evaluation.batches > 0 and n_batches % cfg.evaluation.batches == 0
            ):
                evaluation_metrics = evaluate_model(evaluation_loaders, model, evaluation_results_tracker, logger, batch_header)
                batch_header, evaluated_this_batch = None, True

            # SAVE MODEL CHECKPOINT
            if isinstance(cfg.save_ckpt.batches, int) and (
                # Save checkpoint after every `cfg.save_ckpt.batches` batches trained
                cfg.save_ckpt.batches > 0 and n_batches % cfg.save_ckpt.batches == 0
            ):
                ckpt_file_name = f"ckpt_batch-{n_batches}.pth"
                _ = save_checkpoint(logger, model.state_dict(), ckpt_file_name, batch_header)
                batch_header = None

            # Empty line in case anything logged for this batch
            if batch_header is None:
                logger.log("", with_time=False)

            # Break if completed training required number of batches
            if isinstance(cfg.training.stop.batches, int) and n_batches >= cfg.training.stop.batches:
                break

        # UPDATE LEARNING RATE
        if isinstance(cfg.scheduler.step.epochs, int) and (
            # Update learning rate every `cfg.scheduler.step.epochs` epochs trained
            cfg.scheduler.step.epochs > 0 and n_epochs % cfg.scheduler.step.epochs == 0
        ):
            scheduler.step()

        # LOG EPOCH METRICS
        if isinstance(cfg.training.log.epochs, int) and (
            # Log aggregated training metrics every `cfg.training.log.epochs` epochs trained
            cfg.training.log.epochs > 0 and n_epochs % cfg.training.log.epochs == 0
        ):
            train_metrics = [("Training", training_results_tracker.compute())]
            _ = log_metrics(logger, train_metrics, epoch_header)
            epoch_header = None

        # EVALUATE MODEL
        if isinstance(cfg.evaluation.epochs, int) and (
            # Evaluate after every `cfg.evaluation.epochs` epochs trained
            cfg.evaluation.epochs > 0 and n_epochs % cfg.evaluation.epochs == 0 or
            # Evaluate at the end of training
            cfg.evaluation.epochs == -1 and n_epochs == cfg.training.stop.epochs
        ):
            if evaluated_this_batch:
                log_metrics(logger, evaluation_metrics, epoch_header)
            else:
                _ = evaluate_model(evaluation_loaders, model, evaluation_results_tracker, logger, epoch_header)
            epoch_header = None

        # SAVE MODEL CHECKPOINT
        if isinstance(cfg.save_ckpt.epochs, int) and (
            # Save checkpoint after every `cfg.save_ckpt.epochs` epochs trained
            cfg.save_ckpt.epochs > 0 and n_epochs % cfg.save_ckpt.epochs == 0 or
            # Save checkpoint at the end of training
            cfg.save_ckpt.epochs == -1 and n_epochs == cfg.training.stop.epochs
        ):
            ckpt_file_name = f"ckpt_epoch-{n_epochs}.pth"
            _ = save_checkpoint(logger, model.state_dict(), ckpt_file_name, epoch_header)
            epoch_header = None

        # Empty line in case anything logged for this epoch
        if epoch_header is None:
            logger.log("", with_time=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--architecture", type=str, required=True, help="Architecture name")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer name")
    parser.add_argument("--scheduler", type=str, default="NoSchedule", help="Scheduler name")
    parser.add_argument("--device_index", type=int, help="GPU device index")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options")
    args = parser.parse_args()

    main(args)