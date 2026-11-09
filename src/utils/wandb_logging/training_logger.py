import logging

import torch
from typing import Dict, Any, Optional


def log_ddp_training_config(
    logger: logging.Logger,
    args,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    """Log experiment header and key configuration flags at DDP training startup.

    Should be called once from the main process only.
    """
    logger.info("Starting training dual projector")
    logger.info(f"Running experiment: '{args.experiment_name}'")

    if world_size > 1:
        logger.info(f"Distributed training enabled: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        logger.info("Single process training (DDP not enabled)")

    logger.info("NOTE: " + ("NO DECODER LOSS APPLIED IN THIS RUN (localization-only mode)" if not args.use_decoder else "Applying decoder loss and generating predictions for answers"))
    logger.info("NOTE: " + ("Question Prior Loss is applied" if args.lambda_q_prior > 0 else "NO QUESTION PRIOR LOSS APPLIED"))
    if args.debug:
        logger.info("NOTE: Debug mode enabled - predictions will be logged")

    logger.info(f"IoU loss type set to: {args.iou_loss_type}")
    logger.info(f"   - Lambda iou:     {args.lambda_iou}")
    logger.info(f"   - Lambda size:    {args.lambda_size}")
    logger.info(f"   - Lambda center:  {args.lambda_center}")
    logger.info(f"   - Lambda aspect:  {args.lambda_aspect}")
    logger.info(f"   - Lambda edges:   {args.lambda_edges}")
    logger.info(f"   - Lambda area:    {args.lambda_area}")

    logger.info("Initial size of answer projector:")
    logger.info(f"   - width:  {args.a_init_w}")
    logger.info(f"   - height: {args.a_init_h}")


def log_training_step_metrics(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    box_pred: torch.Tensor,
    q_mask: torch.Tensor,
    decoder_output: Optional[Any] = None,
    global_step: int = 0,
    epoch: int = 0,
) -> Dict[str, Any]:
    """
    Compute gradient norms and activation statistics for logging to WandB.

    Args:
        model: The CoExVQA model
        outputs: Dict containing model outputs (from forward pass)
        box_pred: Predicted bounding boxes (B, 4)
        q_mask: Question mask (B, seq_len, 1)
        decoder_output: Decoder output object (can be None)
        global_step: Current global training step
        epoch: Current epoch number

    Returns:
        Dict containing all metrics ready to log to wandb via run.log()
    """
    metrics = {}

    # Compute gradient norms
    grad_norms = compute_gradient_norms(model)
    metrics.update(grad_norms)

    # Compute activation statistics
    activation_stats = compute_activation_stats(
        outputs=outputs,
        box_pred=box_pred,
        q_mask=q_mask,
        decoder_output=decoder_output
    )
    metrics.update(activation_stats)

    # Add metadata
    metrics['step'] = global_step
    metrics['epoch'] = epoch

    return metrics


def compute_gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for different components of the model.

    Args:
        model: The CoExVQA model (can be wrapped in DDP)

    Returns:
        Dict mapping metric names to gradient norm values
    """
    grad_norms = {}

    # Unwrap DDP if necessary to access model attributes
    model_unwrapped = model.module if hasattr(model, 'module') else model

    # Backbone encoder gradients
    encoder_grad_norm = 0.0
    encoder_param_count = 0
    for name, param in model_unwrapped.backbone_model.encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            encoder_grad_norm += grad_norm ** 2
            encoder_param_count += 1
    if encoder_param_count > 0:
        grad_norms['grad_norm/encoder'] = (encoder_grad_norm ** 0.5) / encoder_param_count

    # Backbone decoder gradients
    decoder_grad_norm = 0.0
    decoder_param_count = 0
    for name, param in model_unwrapped.backbone_model.decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            decoder_grad_norm += grad_norm ** 2
            decoder_param_count += 1
    if decoder_param_count > 0:
        grad_norms['grad_norm/decoder'] = (decoder_grad_norm ** 0.5) / decoder_param_count

    # Question projector gradients
    q_proj_grad_norm = 0.0
    q_proj_param_count = 0
    for name, param in model_unwrapped.q_proj.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            q_proj_grad_norm += grad_norm ** 2
            q_proj_param_count += 1
    if q_proj_param_count > 0:
        grad_norms['grad_norm/q_proj'] = (q_proj_grad_norm ** 0.5) / q_proj_param_count

    # Answer projector gradients
    a_proj_grad_norm = 0.0
    a_proj_param_count = 0
    for name, param in model_unwrapped.a_proj.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            a_proj_grad_norm += grad_norm ** 2
            a_proj_param_count += 1
    if a_proj_param_count > 0:
        grad_norms['grad_norm/a_proj'] = (a_proj_grad_norm ** 0.5) / a_proj_param_count

    # Gating module gradients (if exists)
    if hasattr(model_unwrapped, 'gate') and model_unwrapped.gate is not None:
        gating_grad_norm = 0.0
        gating_param_count = 0
        for name, param in model_unwrapped.gate.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gating_grad_norm += grad_norm ** 2
                gating_param_count += 1
        if gating_param_count > 0:
            grad_norms['grad_norm/gating'] = (gating_grad_norm ** 0.5) / gating_param_count

    return grad_norms


def compute_activation_stats(
    outputs: Dict[str, Any],
    box_pred: torch.Tensor,
    q_mask: torch.Tensor,
    decoder_output: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute activation statistics for model outputs.

    Args:
        outputs: Dict containing model outputs (encoder_outputs, encoder_attention_mask, etc.)
        box_pred: Predicted bounding boxes (B, 4)
        q_mask: Question mask (B, seq_len, 1)
        decoder_output: Decoder output object (can be None)

    Returns:
        Dict mapping metric names to activation statistics
    """
    activation_stats = {}

    # Box prediction statistics
    activation_stats['activations/box_pred_mean'] = box_pred.mean().item()
    activation_stats['activations/box_pred_std'] = box_pred.std().item()
    activation_stats['activations/box_pred_min'] = box_pred.min().item()
    activation_stats['activations/box_pred_max'] = box_pred.max().item()

    # Question mask statistics
    activation_stats['activations/q_mask_mean'] = q_mask.mean().item()
    activation_stats['activations/q_mask_std'] = q_mask.std().item()
    activation_stats['activations/q_mask_sparsity'] = (q_mask < 0.1).float().mean().item()

    # Decoder input statistics
    if decoder_output is not None:
        decoder_hidden = outputs["encoder_outputs"].last_hidden_state
        activation_stats['activations/decoder_input_mean'] = decoder_hidden.mean().item()
        activation_stats['activations/decoder_input_std'] = decoder_hidden.std().item()
        activation_stats['activations/decoder_input_norm'] = decoder_hidden.norm().item() / (decoder_hidden.numel() ** 0.5)

        # Attention mask statistics (crucial for diagnosing token pruning issues)
        encoder_attn_mask = outputs["encoder_attention_mask"]
        activation_stats['activations/encoder_attn_mask_mean'] = encoder_attn_mask.float().mean().item()
        activation_stats['activations/num_attended_tokens'] = encoder_attn_mask.sum(dim=1).float().mean().item()
        activation_stats['activations/seq_len'] = encoder_attn_mask.shape[1]

        # Check if embeddings have collapsed (all similar)
        # Compare first token embedding to second token embedding
        pairwise_sim = torch.cosine_similarity(
            decoder_hidden[:, :1, :],
            decoder_hidden[:, 1:2, :] if decoder_hidden.shape[1] > 1 else decoder_hidden[:, :1, :],
            dim=-1
        ).mean().item()
        activation_stats['activations/embedding_collapse_metric'] = pairwise_sim

    # Decoder output statistics
    if decoder_output is not None and decoder_output.logits is not None:
        logits = decoder_output.logits
        activation_stats['activations/decoder_logits_mean'] = logits.mean().item()
        activation_stats['activations/decoder_logits_std'] = logits.std().item()
        activation_stats['activations/decoder_logits_max'] = logits.max().item()

        # Entropy of predictions (measure of confidence)
        # High entropy = uncertain, low entropy = confident
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        activation_stats['activations/decoder_entropy'] = entropy

    return activation_stats
