"""Average the last N step checkpoints of a run into one model (paper sec. 6.1).

Training already writes averaged.pt when save_every_steps is set; use this to
average a different number, e.g. the paper's big model used the last 20:

    python scripts/average_checkpoints.py checkpoints/00_paper_base --last 5
    python evaluate.py --config configs/exp/00_paper_base.yaml --checkpoint checkpoints/00_paper_base/averaged.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.checkpoints import average_checkpoints, step_checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: <ckpt_dir>/averaged.pt")
    args = ap.parse_args()

    paths = step_checkpoints(args.ckpt_dir)[-args.last:]
    if len(paths) < args.last:
        print(f"Only {len(paths)} step checkpoints in {args.ckpt_dir} (asked for {args.last})")
    avg = average_checkpoints(paths)
    out = Path(args.out or Path(args.ckpt_dir) / "averaged.pt")
    torch.save(avg, out)
    print(f"Averaged {', '.join(avg['averaged_from'])} -> {out}")


if __name__ == "__main__":
    main()
