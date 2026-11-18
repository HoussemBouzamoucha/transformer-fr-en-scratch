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

from src.data.dataset import Multi30KDataset, collate_fn
from src.model.transformer import Transformer
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.scheduler import InverseSqrtScheduler
from src.tokenizer.loader import load_tokenizer
from src.eval.metrics import evaluate_bleu
from src.utils import set_seed, load_config


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", f"checkpoints/{Path(config_path).stem}"))

    # Data
    data_dir = cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}")
    max_pairs = cfg["data"].get("max_train_pairs", None)
    train_set = Multi30KDataset("train", data_dir, tokenizer, max_len=cfg["data"]["max_tokens"], max_pairs=max_pairs)
    val_set   = Multi30KDataset("val",   data_dir, tokenizer, max_len=cfg["data"]["max_tokens"])
    # Raw validation sentences for BLEU (scored against the untouched references).
    bleu_n = cfg.get("logging", {}).get("bleu_sentences", 500)
    with open(Path(data_dir) / "val.fr", encoding="utf-8") as f:
        val_src = [l.strip() for l in f][:bleu_n]
    with open(Path(data_dir) / "val.en", encoding="utf-8") as f:
        val_ref = [l.strip() for l in f][:bleu_n]
    restrict = cfg.get("inference", {}).get("restrict_target_vocab", True)

    collate = partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Model
    m = cfg["model"]
    model = Transformer(
        vocab_size       = vocab_size,
        d_model          = m["d_model"],
        n_heads          = m["n_heads"],
        n_encoder_layers = m["n_encoder_layers"],
        n_decoder_layers = m["n_decoder_layers"],
        d_ff             = m["d_ff"],
        max_len          = m["max_seq_len"],
        dropout          = m["dropout"],
        pad_id           = pad_id,
        tie_embeddings   = m.get("shared_vocab", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = LabelSmoothedCrossEntropy(
        vocab_size,
        cfg["training"]["label_smoothing"],
        pad_id
    )
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = InverseSqrtScheduler(optimizer, m["d_model"], cfg["training"]["warmup_steps"])
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg["training"]["fp16"] and device.type == "cuda")

    best_val_loss = float("inf")
    step = 0
    # Timing for the tokenizer benchmark. train_seconds counts only the training
    # loop (no validation / BLEU decoding); the epoch-end .item() syncs the GPU.
    t_start = time.time()
    train_seconds = 0.0
    tokens_seen = {"src": 0, "tgt": 0}
    best = {}

    for epoch in range(1, 1000):
        model.train()
        # Accumulate on the GPU; converting to Python floats every step (.item())
        # forces a CPU-GPU sync and stalls the launch queue.
        total_loss = torch.zeros((), device=device)
        total_tokens = torch.zeros((), device=device)
        src_tokens = torch.zeros((), device=device)
        t_epoch = time.time()

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    tgt_out.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            # Set the LR for this step *before* the update, so step 1 uses the
            # warmup LR instead of Adam's default 1e-3.
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            n_tokens = (tgt_out != pad_id).sum()
            total_loss   += loss.detach().float() * n_tokens
            total_tokens += n_tokens
            src_tokens   += (src != pad_id).sum()

            if step >= cfg["training"]["max_steps"]:
                break

        train_loss = (total_loss / total_tokens).item()
        train_seconds += time.time() - t_epoch
        tokens_seen["tgt"] += int(total_tokens.item())
        tokens_seen["src"] += int(src_tokens.item())

        # Validation
        model.eval()
        val_loss = torch.zeros((), device=device)
        val_tokens = torch.zeros((), device=device)
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                tgt_in  = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                with torch.amp.autocast('cuda', enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                    logits = model(src, tgt_in)
                    loss = criterion(
                        logits.reshape(-1, vocab_size),
                        tgt_out.reshape(-1)
                    )
                n_tokens = (tgt_out != pad_id).sum()
                val_loss   += loss.float() * n_tokens
                val_tokens += n_tokens

        val_loss = (val_loss / val_tokens).item()
        bleu_txt = ""
        if bleu_n:
            res = evaluate_bleu(model, tokenizer, val_src, val_ref, device,
                                max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"],
                                restrict_output=restrict)
            bleu_txt = f" | val BLEU {res['bleu']:.2f} chrF {res['chrf']:.1f}"
        print(f"Epoch {epoch:2d} | step {step:6d} | lr {scheduler.get_lr():.6f} "
              f"| train loss {train_loss:.4f} | val loss {val_loss:.4f}{bleu_txt}")
        if bleu_n:
            for src_l, hyp, ref in list(zip(val_src, res["hypotheses"], val_ref))[:3]:
                print(f"    FR  {src_l}\n    OUT {hyp}\n    REF {ref}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_bleu": res["bleu"] if bleu_n else None,
                "config": cfg,
                # Ids only mean something with this exact vocab; evaluate.py checks it.
                "tokenizer": tokenizer.fingerprint(),
                "train_seconds": train_seconds,
            }, ckpt_dir / "best.pt")
            best = {"best_step": step, "best_epoch": epoch, "best_val_loss": val_loss,
                    "best_val_bleu": res["bleu"] if bleu_n else None,
                    "train_seconds_to_best": train_seconds}
            print(f"  → saved {ckpt_dir / 'best.pt'} (val loss {val_loss:.4f})")

        if step >= cfg["training"]["max_steps"]:
            print("Reached max steps.")
            break

    summary = {
        "config": str(config_path), "seed": cfg["seed"], "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": n_params, "vocab_size": vocab_size, "steps": step, "epochs": epoch,
        "train_seconds": train_seconds, "total_seconds": time.time() - t_start,
        "steps_per_second": step / train_seconds if train_seconds else None,
        "src_tokens_per_second": tokens_seen["src"] / train_seconds if train_seconds else None,
        "tgt_tokens_per_second": tokens_seen["tgt"] / train_seconds if train_seconds else None,
        **best,
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training loop {train_seconds / 60:.1f} min ({summary['steps_per_second'] or 0:.2f} steps/s), "
          f"best step {best.get('best_step')} -> {ckpt_dir / 'train_summary.json'}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/thin_slice.yaml"
    train(config)