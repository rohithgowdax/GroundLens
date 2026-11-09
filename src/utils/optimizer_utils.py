import logging
from torch.optim import AdamW, RAdam, SGD, Optimizer


def setup_optimizer(
    model,
    optimizer_name: str,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    logger: logging.Logger
) -> Optimizer:
    """
    Setup optimizer with separate learning rates for backbone and other parameters.

    Args:
        model: Model with parameters (unwrapped, not DDP-wrapped)
        optimizer_name: Name of optimizer ("adamw", "radam", "sgd")
        lr: Learning rate for non-backbone parameters
        backbone_lr: Learning rate for backbone parameters
        weight_decay: Weight decay coefficient
        logger: Logger instance

    Returns:
        Configured optimizer instance
    """
    logger.info(f"Setting up optimizer: {optimizer_name}")

    # Separate backbone parameters from the rest
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone_model."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    backbone_param_count = sum(p.numel() for p in backbone_params)
    other_param_count = sum(p.numel() for p in other_params)
    logger.info(f"Backbone parameters: {backbone_param_count:,} (lr={backbone_lr})")
    logger.info(f"Other parameters:    {other_param_count:,} (lr={lr})")

    # Create parameter groups with different learning rates (skip empties)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    else:
        logger.warning("Backbone param group is empty (no trainable params matched 'backbone_model.').")

    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    else:
        logger.warning("Other param group is empty (no trainable params outside backbone).")

    # Create optimizer
    opt = optimizer_name.lower()
    if opt == "adamw":
        optimizer = AdamW(param_groups, betas=(0.9, 0.95), weight_decay=weight_decay)
    elif opt == "radam":
        optimizer = RAdam(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer
