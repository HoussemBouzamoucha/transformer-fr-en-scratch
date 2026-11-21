import argparse
import json
import random
import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from src.data.dataset import Multi30KDataset, TokenBatchSampler, collate_fn
from src.model.transformer import build_model, describe_architecture
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.scheduler import InverseSqrtScheduler
from src.training.checkpoints import save_step_checkpoint, step_checkpoints, average_checkpoints
from src.tokenizer.loader import load_tokenizer
from src.eval.metrics import evaluate_bleu
from src.utils import set_seed, load_config


def train(config_path: str, seed: int = None, max_steps: int = None, checkpoint_dir: str = None,
          resume: bool = False):
    """Train the model a config describes. seed / max_steps / checkpoint_dir override
    the config (the checkpoints store the config with the overrides applied).
    resume: continue from <checkpoint_dir>/last.pt if it exists (written every
    save_every_steps): same weights, optimizer, LR schedule, fp16 scaler, RNG
    states and position in the epoch, so the run continues where it stopped."""
    cfg = load_config(config_path)
    if seed is not None:
        cfg["seed"] = seed
    if max_steps is not None:
        cfg["training"]["max_steps"] = max_steps
    if checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    elif seed is not None and "checkpoint_dir" not in cfg["training"]:
        cfg["training"]["checkpoint_dir"] = f"checkpoints/{Path(config_path).stem}_seed{seed}"
    set_seed(cfg["seed"])
    tcfg = cfg["training"]
    lcfg = cfg.get("logging", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    use_amp = tcfg["fp16"] and device.type == "cuda"

    # Tokenizer
    tokenizer = load_tokenizer(cfg)
    pad_id = tokenizer.pad_id
    # The real vocab can be smaller than the configured target (BPE stops early
    # when no pair occurs twice), so size the model from the tokenizer itself.
    vocab_size = tokenizer.vocab_size
    if vocab_size != cfg["tokenizer"]["vocab_size"]:
        print(f"Note: tokenizer vocab is {vocab_size}, config says {cfg['tokenizer']['vocab_size']}; using {vocab_size}")
    print(f"Tokenizer: {cfg['tokenizer'].get('type', 'sentencepiece')}, vocab {vocab_size}")

    # One folder per config, so runs don't overwrite each other's best.pt.
    ckpt_dir = Path(tcfg.get("checkpoint_dir", f"checkpoints/{Path(config_path).stem}"))

    # Data
    data_dir = cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}")
    max_pairs = cfg["data"].get("max_train_pairs", None)
    train_set = Multi30KDataset("train", data_dir, tokenizer, max_len=cfg["data"]["max_tokens"], max_pairs=max_pairs)
    val_set   = Multi30KDataset("val",   data_dir, tokenizer, max_len=cfg["data"]["max_tokens"])
    # Raw validation sentences for BLEU (scored against the untouched references).
    bleu_n = lcfg.get("bleu_sentences", 500)
    with open(Path(data_dir) / "val.fr", encoding="utf-8") as f:
        val_src = [l.strip() for l in f][:bleu_n]
    with open(Path(data_dir) / "val.en", encoding="utf-8") as f:
        val_ref = [l.strip() for l in f][:bleu_n]
    restrict = cfg.get("inference", {}).get("restrict_target_vocab", True)

    collate = partial(collate_fn, pad_id=pad_id)
    # Own generator: each epoch's DataLoader iterator draws its worker seed from it
    # instead of the global RNG, so the global RNG (dropout) only advances with
    # training steps, and a resumed run gets exactly the same dropout masks.
    loader_kw = dict(collate_fn=collate, num_workers=2, pin_memory=True, prefetch_factor=2,
                     generator=torch.Generator().manual_seed(cfg["seed"]))
    # batching: sentences = fixed number of pairs per batch (the old default);
    # tokens = length-bucketed batches of ~batch_tokens tokens (paper sec. 5.1).
    batching = tcfg.get("batching", "sentences")
    sampler = None
    if batching == "tokens":
        sampler = TokenBatchSampler.from_dataset(train_set, tcfg["batch_tokens"], seed=cfg["seed"])
        train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_kw)
    elif batching == "sentences":
        train_loader = DataLoader(train_set, batch_size=tcfg.get("batch_size", 128), shuffle=True, **loader_kw)
    else:
        raise ValueError(f"training.batching must be 'sentences' or 'tokens', got {batching!r}")
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, **loader_kw)

    # Gradients of accum_steps batches are summed before each optimizer step, so
    # one GPU sees the batch size of the paper's 8 GPUs.
    accum = tcfg.get("accum_steps", 1)
    grad_clip = tcfg.get("grad_clip")  # null = no clipping (the paper doesn't clip)
    eval_every = lcfg.get("eval_every_steps")  # null = evaluate at the end of every epoch
    save_every = tcfg.get("save_every_steps")  # null = no step checkpoints / averaging
    keep_last = tcfg.get("keep_last", 5)
    per_batch = tcfg["batch_tokens"] if sampler else tcfg.get("batch_size", 128)
    print(f"Batching: {batching} ({per_batch} per batch) x {accum} accumulation "
          f"| {len(train_loader)} batches/epoch")

    # Model
    m = cfg["model"]
    model = build_model(m, vocab_size, pad_id).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} | {describe_architecture(m)}")

    # Loss, optimizer, scheduler (paper sec. 5.3-5.4)
    criterion = LabelSmoothedCrossEntropy(vocab_size, tcfg["label_smoothing"], pad_id)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = InverseSqrtScheduler(optimizer, m["d_model"], tcfg["warmup_steps"])
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # eval_seconds: time spent validating / decoding / writing checkpoints, which
    # train_seconds (used by the tokenizer benchmark) leaves out.
    state = {"step": 0, "best_val_loss": float("inf"), "best": {}, "eval_seconds": 0.0}
    t_start = time.time()
    tokens_seen = {"src": 0, "tgt": 0}
    # Train-loss accumulators since the last report; kept on the GPU because
    # .item() every step forces a CPU-GPU sync and stalls the launch queue.
    run = {}

    def reset_running():
        run["loss"] = torch.zeros((), device=device)
        run["tgt"] = torch.zeros((), device=device)
        run["src"] = torch.zeros((), device=device)

    def ckpt_payload(epoch, **extra):
        return {
            "epoch": epoch, "step": state["step"], "model_state": model.state_dict(),
            "config": cfg,
            # Ids only mean something with this exact vocab; evaluate.py checks it.
            "tokenizer": tokenizer.fingerprint(), **extra,
        }

    def train_step(group):
        """One optimizer step over a group of up to `accum` batches."""
        optimizer.zero_grad(set_to_none=True)
        # Weight each batch by its share of target tokens so the update equals
        # the token-mean loss over the whole group.
        n_group = sum(int((tgt[:, 1:] != pad_id).sum()) for _, tgt in group)
        for src, tgt in group:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            n_tokens = (tgt_out != pad_id).sum()
            scaler.scale(loss * (n_tokens / n_group)).backward()
            run["loss"] += loss.detach().float() * n_tokens
            run["tgt"] += n_tokens
            run["src"] += (src != pad_id).sum()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Set the LR for this step *before* the update, so step 1 uses the
        # warmup LR instead of Adam's default 1e-3.
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()
        state["step"] += 1

    def evaluate(epoch):
        t_eval = time.time()
        train_loss = (run["loss"] / run["tgt"]).item() if run["tgt"] > 0 else float("nan")
        tokens_seen["tgt"] += int(run["tgt"].item())
        tokens_seen["src"] += int(run["src"].item())
        reset_running()

        model.eval()
        val_loss = torch.zeros((), device=device)
        val_tokens = torch.zeros((), device=device)
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(src, tgt_in)
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                n_tokens = (tgt_out != pad_id).sum()
                val_loss   += loss.float() * n_tokens
                val_tokens += n_tokens
        val_loss = (val_loss / val_tokens).item()

        bleu_txt, res = "", None
        if bleu_n:
            # Greedy here for speed; evaluate.py uses beam search (inference.beam_size).
            res = evaluate_bleu(model, tokenizer, val_src, val_ref, device,
                                max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"],
                                restrict_output=restrict)
            bleu_txt = f" | val BLEU (greedy) {res['bleu']:.2f} chrF {res['chrf']:.1f}"
        print(f"Epoch {epoch:2d} | step {state['step']:6d} | lr {scheduler.get_lr():.6f} "
              f"| train loss {train_loss:.4f} | val loss {val_loss:.4f}{bleu_txt}")
        if res:
            for src_l, hyp, ref in list(zip(val_src, res["hypotheses"], val_ref))[:3]:
                print(f"    FR  {src_l}\n    OUT {hyp}\n    REF {ref}")

        if val_loss < state["best_val_loss"]:
            state["best_val_loss"] = val_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            train_seconds = t_eval - t_start - state["eval_seconds"]
            torch.save(ckpt_payload(
                epoch, optimizer_state=optimizer.state_dict(), val_loss=val_loss,
                val_bleu=res["bleu"] if res else None, train_seconds=train_seconds,
            ), ckpt_dir / "best.pt")
            state["best"] = {"best_step": state["step"], "best_epoch": epoch, "best_val_loss": val_loss,
                             "best_val_bleu": res["bleu"] if res else None,
                             "train_seconds_to_best": train_seconds}
            print(f"  → saved {ckpt_dir / 'best.pt'} (val loss {val_loss:.4f})")
        model.train()
        state["eval_seconds"] += time.time() - t_eval

    last_path = ckpt_dir / "last.pt"

    def save_last(epoch, batches_in_epoch):
        """Everything needed to continue this run exactly (see --resume)."""
        payload = ckpt_payload(
            epoch, batches_in_epoch=batches_in_epoch,
            optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(),
            scheduler_step=scheduler.step_num, train_state=dict(state),
            tokens_seen={"src": tokens_seen["src"] + int(run["src"].item()),
                         "tgt": tokens_seen["tgt"] + int(run["tgt"].item())},
            elapsed=time.time() - t_start,
            rng={"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(),
                 "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = last_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, last_path)  # never leaves a half-written last.pt behind

    def after_step(epoch, batches_in_epoch):
        step = state["step"]
        if save_every and step % save_every == 0:
            t_save = time.time()
            save_step_checkpoint(ckpt_dir, step, ckpt_payload(epoch), keep_last)
            save_last(epoch, batches_in_epoch)
            state["eval_seconds"] += time.time() - t_save
        if eval_every and step % eval_every == 0:
            evaluate(epoch)

    reset_running()
    max_steps = tcfg["max_steps"]
    start_epoch, skip = 1, 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        scheduler.step_num = ck["scheduler_step"]
        state.update(ck["train_state"])
        tokens_seen.update(ck["tokens_seen"])
        t_start = time.time() - ck["elapsed"]
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if ck["rng"]["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])
        start_epoch, skip = ck["epoch"], ck["batches_in_epoch"]
        if sampler is None and skip:
            print("Note: sentence batching can't skip to the saved position; "
                  f"epoch {start_epoch} restarts from its first batch")
            skip = 0
        print(f"Resumed from {last_path}: step {state['step']}, epoch {start_epoch}, "
              f"{skip} batches into it")
    elif resume:
        print(f"No {last_path} yet: starting from scratch")

    epoch = start_epoch
    for epoch in range(start_epoch, 100000):
        if state["step"] >= max_steps:
            break
        model.train()
        seen = skip if epoch == start_epoch else 0  # batches of this epoch already trained on
        if sampler is not None:
            sampler.set_epoch(epoch)
            sampler.skip = seen
        group = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=bool(eval_every)):
            group.append(batch)
            seen += 1
            if len(group) == accum:
                train_step(group)
                group = []
                after_step(epoch, seen)
                if state["step"] >= max_steps:
                    break
        if group and state["step"] < max_steps:  # leftover batches at the end of the epoch
            train_step(group)
            after_step(epoch, seen)

        if not eval_every:
            evaluate(epoch)
        if state["step"] >= max_steps:
            print("Reached max steps.")
            break

    if eval_every and state["step"] % eval_every:
        evaluate(epoch)  # the steps after the last scheduled evaluation
    train_seconds = time.time() - t_start - state["eval_seconds"]

    # Paper sec. 6.1: average the last checkpoints into one model.
    averaged = None
    if save_every:
        if state["step"] % save_every:
            save_step_checkpoint(ckpt_dir, state["step"], ckpt_payload(epoch), keep_last)
        paths = step_checkpoints(ckpt_dir)[-keep_last:]
        avg = average_checkpoints(paths)
        averaged = ckpt_dir / "averaged.pt"
        torch.save(avg, averaged)
        print(f"Averaged {len(paths)} checkpoints ({', '.join(avg['averaged_from'])}) -> {averaged}")

    step = state["step"]
    summary = {
        "config": str(config_path), "seed": cfg["seed"], "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "architecture": describe_architecture(m),
        "parameters": n_params, "vocab_size": vocab_size, "steps": step, "epochs": epoch,
        "train_seconds": train_seconds, "total_seconds": time.time() - t_start,
        "steps_per_second": step / train_seconds if train_seconds else None,
        "src_tokens_per_second": tokens_seen["src"] / train_seconds if train_seconds else None,
        "tgt_tokens_per_second": tokens_seen["tgt"] / train_seconds if train_seconds else None,
        "averaged_checkpoint": str(averaged) if averaged else None,
        **state["best"],
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training loop {train_seconds / 60:.1f} min ({summary['steps_per_second'] or 0:.2f} steps/s), "
          f"best step {state['best'].get('best_step')} -> {ckpt_dir / 'train_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Transformer a config describes.")
    ap.add_argument("config", nargs="?", default="configs/thin_slice.yaml")
    ap.add_argument("--seed", type=int, default=None, help="override the config's seed")
    ap.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: training.checkpoint_dir, else checkpoints/<config>[_seed<N>]")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/last.pt")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # "→" in the log, also on a Windows console/file
    train(a.config, seed=a.seed, max_steps=a.max_steps, checkpoint_dir=a.checkpoint_dir,
          resume=a.resume)
