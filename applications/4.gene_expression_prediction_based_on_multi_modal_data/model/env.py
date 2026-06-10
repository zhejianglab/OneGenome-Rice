# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Runtime environment defaults, logging, and seeding for training."""

import logging
import os
import random

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def configure_runtime() -> None:
    """Idempotent hook for entry scripts; env vars and seeds are set at import time."""
    pass
