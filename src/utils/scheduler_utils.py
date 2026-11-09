import logging
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from transformers import get_scheduler


def setup_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    logger: logging.Logger
):
    """
    Setup learning rate scheduler based on configuration.

    Args:
        scheduler_name: Name of scheduler ("cosine", "constant_with_warmup", "onecycle", "reduceonplateau")
        optimizer: PyTorch optimizer instance
        epochs: Number of training epochs
        steps_per_epoch: Number of batches per epoch
        logger: Logger instance for logging

    Returns:
        tuple: (scheduler, scheduler_step_on)
            - scheduler: Configured scheduler instance
            - scheduler_step_on: When to step ("batch" or "epoch")
    """
    logger.info(f"Setting up scheduler: {scheduler_name}")
    num_training_steps = epochs * max(1, steps_per_epoch)
    scheduler_step_on = None

    if scheduler_name == "cosine":
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=int(0.10 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        scheduler_step_on = "batch"
    elif scheduler_name == "constant_with_warmup":
        scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=int(0.10 * num_training_steps),
        )
        scheduler_step_on = "batch"
    elif scheduler_name == "onecycle":
        # OneCycleLR (step each batch)
        max_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            steps_per_epoch=max(1, steps_per_epoch),
            epochs=epochs,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
            three_phase=False,
        )
        scheduler_step_on = "batch"
    elif scheduler_name == "reduceonplateau":
        # ReduceLROnPlateau
        min_lrs = [g["lr"] * 1e-3 for g in optimizer.param_groups]
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            threshold=1e-4,
            cooldown=1,
            min_lr=min_lrs,
        )
        scheduler_step_on = "epoch"
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler, scheduler_step_on
