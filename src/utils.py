import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode is for debugging/reproducibility runs, not for timing.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)