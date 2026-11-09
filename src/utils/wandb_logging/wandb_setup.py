import wandb
import time
import os
from typing import Optional


def setup_wandb(config_dict: dict, project_name="GroundLens", prefix="groundlens-train", run_id: Optional[str] = None) -> wandb.sdk.wandb_run.Run:
    """
    Creates *or resumes* a W&B run. If run_id is provided, we resume that exact run.
    We always use the same config for visibility, and we keep silent console as before.
    """
    # silence mode
    os.environ.setdefault("WANDB_SILENT", "true")

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    # If a run id is given, then resumes the same run, else it does not 
    if run_id is None:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{prefix}-{current_time}"
        run = wandb.init(
            project=project_name,
            entity="kjetiki-university-of-oslo",
            name=run_name,
            config=config_dict,
            reinit=True,
            settings=wandb.Settings(silent=True, _disable_stats=True),
        )
    else:
        # resume same run
        run = wandb.init(
            project=project_name,
            entity="kjetiki-university-of-oslo",
            id=run_id,
            resume="must",
            config=config_dict,
            reinit=True,
            settings=wandb.Settings(silent=True, _disable_stats=True),
        )
    return run