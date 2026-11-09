import os
import sys
import logging


logger = logging.getLogger(__name__)


def get_hf_token():
    """Get HuggingFace token from environment variable HF_TOKEN."""
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        logger.error("HuggingFace token not found in environment (HF_TOKEN).")
        sys.exit(1)
    logger.info("HuggingFace token loaded.")
    return hf_token