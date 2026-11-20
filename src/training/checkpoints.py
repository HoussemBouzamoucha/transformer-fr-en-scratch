"""Periodic step checkpoints and checkpoint averaging.

Vaswani et al. 2017 sec. 6.1: "For the base models, we used a single model
obtained by averaging the last 5 checkpoints, which were written at 10-minute
intervals." We write them every `save_every_steps` optimizer steps instead of
every 10 minutes, so runs on different GPUs average the same steps.
"""
import re
from pathlib import Path

import torch

STEP_RE = re.compile(r"step_(\d+)\.pt$")


def step_checkpoints(ckpt_dir):
    """Step checkpoints in ckpt_dir, oldest first."""
    found = [(int(m.group(1)), p) for p in Path(ckpt_dir).glob("step_*.pt")
             if (m := STEP_RE.search(p.name))]
    return [p for _, p in sorted(found)]


def save_step_checkpoint(ckpt_dir, step, payload, keep_last):
    """Write step_<step>.pt (model weights only) and delete all but the newest keep_last."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save(payload, path)
    for old in step_checkpoints(ckpt_dir)[:-keep_last]:
        old.unlink()
    return path


def average_checkpoints(paths):
    """Element-wise mean of the model weights of several checkpoints.

    Returns the newest checkpoint's dict with model_state replaced by the mean,
    so config / tokenizer fingerprint / step carry over and evaluate.py can load it.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no checkpoints to average")
    total = None
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)["model_state"]
        if total is None:
            total = {k: v.double() if v.is_floating_point() else v.clone() for k, v in state.items()}
            ref = state
        else:
            if state.keys() != total.keys():
                raise ValueError(f"{p} has different parameters than {paths[0]}")
            for k, v in state.items():
                if v.is_floating_point():
                    total[k] += v.double()
    avg = {k: (v / len(paths)).to(ref[k].dtype) if v.is_floating_point() else v
           for k, v in total.items()}

    out = torch.load(paths[-1], map_location="cpu", weights_only=False)
    out["model_state"] = avg
    out["averaged_from"] = [p.name for p in paths]
    return out
