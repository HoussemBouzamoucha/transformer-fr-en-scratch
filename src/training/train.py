import json
import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


def train(config_path: str):
    cfg = load_config(config_path)
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
    loader_kw = dict(collate_fn=collate, num_workers=2, pin_memory=True, prefetch_factor=2)
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

    def after_step(epoch):
        step = state["step"]
        if save_every and step % save_every == 0:
            t_save = time.time()
            save_step_checkpoint(ckpt_dir, step, ckpt_payload(epoch), keep_last)
            state["eval_seconds"] += time.time() - t_save
        if eval_every and step % eval_every == 0:
            evaluate(epoch)

    reset_running()
    max_steps = tcfg["max_steps"]
    epoch = 0
    for epoch in range(1, 100000):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        group = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=bool(eval_every)):
            group.append(batch)
            if len(group) == accum:
                train_step(group)
                group = []
                after_step(epoch)
                if state["step"] >= max_steps:
                    break
        if group and state["step"] < max_steps:  # leftover batches at the end of the epoch
            train_step(group)
            after_step(epoch)

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
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/thin_slice.yaml"
    train(config)
