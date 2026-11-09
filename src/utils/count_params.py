def human_count(n: int) -> str:
    if n < 1_000: return f"{n}"
    for unit, div in (("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000)):
        if n < div * 1000:
            return f"{n / div:.2f}{unit}"
    return f"{n / 1_000_000_000:.2f}B"

def count_params(module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters()
               if (p.requires_grad or not trainable_only))

def log_param_block(logger, name: str, module):
    tot = count_params(module, trainable_only=False)
    trn = count_params(module, trainable_only=True)
    logger.info(f"{name:35s} total={tot:,} ({human_count(tot)}), "
                f"trainable={trn:,} ({human_count(trn)})")

def log_param_model(logger, model):
    """
    Log parameter counts for CoExVQA model components.

    Args:
        logger: Logger instance
        model: CoExVQA model (unwrapped, not DDP-wrapped)
    """
    backbone_name = model.config.backbone_model.split("/")[-1] if hasattr(model.config, "backbone_model") and model.config.backbone_model else "Backbone"
    logger.info("Parameter counts:")
    log_param_block(logger, "CoExVQA Total", model)
    log_param_block(logger, f"  Backbone ({backbone_name})", model.backbone_model)

    # Answer heatmap projector blocks
    log_param_block(logger, "  AnswerHeatmapProjector Total", model.a_proj)
    log_param_block(logger, "    A/FusionBlock", model.a_proj.fusion_block)
    log_param_block(logger, "    A/ContextAgg", model.a_proj.context_agg_block)
    log_param_block(logger, "    A/Projector", model.a_proj.projector_block)
    log_param_block(logger, "    A/TaskHead", model.a_proj.task_head)

    # Question heatmap projector blocks
    log_param_block(logger, "  QuestionHeatmapProjector Total", model.q_proj)
    log_param_block(logger, "    Q/FusionBlock", model.q_proj.fusion_block)
    log_param_block(logger, "    Q/ContextAgg", model.q_proj.context_agg_block)
    log_param_block(logger, "    Q/Projector", model.q_proj.projector_block)
    log_param_block(logger, "    Q/TaskHead", model.q_proj.task_head)