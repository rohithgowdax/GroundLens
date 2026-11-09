from typing import Optional
import torch


# Import loss implementations
from src.model.losses.projection_loss import get_projection_loss
from src.model.losses.q_mask_loss import get_q_prior_mse_loss
from src.model.losses.decoder import get_decoder_loss_weight
from src.model.losses.ib_loss import compute_ib_loss


def compute_losses(
    decoder_output,
    box_pred: torch.Tensor,
    q_mask: torch.Tensor,
    gt_boxes_xyxy_abs,
    full_img_sizes: torch.Tensor,
    batch: dict,
    args,
    epoch: int,
    device: torch.device,
    image_embeddings: Optional[torch.Tensor] = None,
    gated_embeddings: Optional[torch.Tensor] = None,
) -> dict:
    """Compute all training losses and return them as a dict.

    Args:
        decoder_output: Decoder output object with a ``.loss`` attribute, or None.
        box_pred: Predicted boxes in normalized cxcywh format, shape (B, 4).
        q_mask: Predicted question mask, shape (B, seq_len, 1).
        gt_boxes_xyxy_abs: GT boxes in absolute xyxy pixel coordinates.
        full_img_sizes: Image sizes (W, H), shape (B, 2).
        batch: Raw batch dict; used to read q_prior fields when applicable.
        args: Parsed argument namespace carrying all lambda weights and flags.
        epoch: Current epoch (1-indexed), used for decoder warmup schedule.
        device: Device for zero-valued placeholder tensors.
        image_embeddings: (B, seq_len, d_model) raw encoder embeddings; only required when
            args.lambda_ib > 0. None otherwise.
        gated_embeddings: (B, seq_len, d_model) gated encoder embeddings; only required when
            args.lambda_ib > 0. None otherwise.

    Returns:
        Dict with keys: ``total``, ``decoder``, ``projection``, ``q_prior``,
        ``ib``, ``ib_comp1``, ``ib_suff``, ``ib_comp2``,
        ``iou``, ``size``, ``center``, ``edges``, ``aspect``, ``area``.
        ``total``, ``decoder``, ``projection``, ``q_prior``, ``ib`` are tensors;
        the remaining values are Python floats.
    """
    # Decoder loss
    if not args.use_decoder or decoder_output is None:
        decoder_loss = torch.tensor(0.0, device=device)
    else:
        decoder_weight = get_decoder_loss_weight(
            current_epoch=epoch - 1,
            warmup_epochs=args.decoder_loss_warmup_epochs,
        )
        decoder_loss = decoder_output.loss * decoder_weight * args.lambda_dec

    # Projection (localization) loss
    proj_loss_dict = get_projection_loss(
        gt_boxes_xyxy_abs=gt_boxes_xyxy_abs,
        pred_boxes_cxcywh_norm=box_pred,
        full_img_sizes=full_img_sizes,
        lambda_iou=args.lambda_iou,
        lambda_size=args.lambda_size,
        lambda_center=args.lambda_center,
        lambda_aspect=args.lambda_aspect,
        lambda_edges=args.lambda_edges,
        lambda_area=args.lambda_area,
        iou_loss_type=args.iou_loss_type,
    )
    proj_loss = proj_loss_dict["loss_total"]

    # Question-prior loss
    if args.lambda_q_prior > 0:
        q_prior_loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask,
            q_prior_patch=batch["q_prior_patch"],
            q_prior_patch_shape=batch["q_prior_patch_shape"],
            q_prior_mask=batch.get("q_prior_mask", None),
        ) * args.lambda_q_prior
    else:
        q_prior_loss = torch.tensor(0.0, device=device)

    # Information Bottleneck loss
    lambda_ib = getattr(args, "lambda_ib", 0.0)
    if lambda_ib > 0 and image_embeddings is not None and gated_embeddings is not None:
        ib_dict = compute_ib_loss(
            image_embeddings=image_embeddings,
            q_mask=q_mask,
            gated_embeddings=gated_embeddings,
            box_pred=box_pred,
            lambda_ib=lambda_ib,
            beta1=getattr(args, "ib_beta1", 1.0),
            beta2=getattr(args, "ib_beta2", 1.0),
        )
        ib_loss = ib_dict["ib_total"]
    else:
        ib_loss = torch.tensor(0.0, device=device)
        ib_dict = {"ib_comp1": 0.0, "ib_suff": 0.0, "ib_comp2": 0.0}

    return {
        "total": decoder_loss + proj_loss + q_prior_loss + ib_loss,
        "decoder": decoder_loss,
        "projection": proj_loss,
        "q_prior": q_prior_loss,
        "ib": ib_loss,
        "ib_comp1": ib_dict["ib_comp1"],
        "ib_suff": ib_dict["ib_suff"],
        "ib_comp2": ib_dict["ib_comp2"],
        "iou": float(proj_loss_dict["loss_iou"]),
        "size": float(proj_loss_dict["loss_size"]),
        "center": float(proj_loss_dict["loss_center"]),
        "edges": float(proj_loss_dict["loss_edges"]),
        "aspect": float(proj_loss_dict["loss_aspect"]),
        "area": float(proj_loss_dict["loss_area"]),
    }
