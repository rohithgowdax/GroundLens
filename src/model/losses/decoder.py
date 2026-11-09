def get_decoder_loss_weight(current_epoch: int, warmup_epochs: int = 0) -> float:
    """
    Get the decoder loss weight for the current training epoch, with optional warmup.

    Args:
        current_epoch: Current training epoch (0-indexed)
        warmup_epochs: Number of epochs to warm up decoder loss from 0 to 1 (0 = no warmup)

    Returns:
        float: Weight for decoder loss in [0, 1]
                - Returns 0.0 at epoch 0 if warmup is enabled
                - Linearly increases to 1.0 over warmup_epochs
                - Returns 1.0 after warmup completes or if warmup_epochs=0
    """
    if warmup_epochs == 0:
        # No warmup: always return 1.0
        return 1.0

    if current_epoch >= warmup_epochs:
        # Warmup complete: return 1.0
        return 1.0

    # Linear warmup: weight = current_epoch / warmup_epochs
    return float(current_epoch) / float(warmup_epochs)