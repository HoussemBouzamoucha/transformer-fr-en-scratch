# main training loopimport sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import sentencepiece as spm
from tqdm import tqdm

from src.data.dataset import Multi30KDataset, collate_fn
from src.model.transformer import Transformer
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.scheduler import InverseSqrtScheduler
from src.utils import set_seed, load_config


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("data/vocab/spm_8k.model")
    pad_id = sp.piece_to_id("<pad>")

    # Data
    train_set = Multi30KDataset("train", "data/raw/multi30k", "data/vocab/spm_8k.model",
                                max_len=cfg["data"]["max_tokens"])
    val_set   = Multi30KDataset("val",   "data/raw/multi30k", "data/vocab/spm_8k.model",
                                max_len=cfg["data"]["max_tokens"])

    collate = partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, collate_fn=collate)
    # Model
    m = cfg["model"]
    model = Transformer(
        vocab_size       = cfg["tokenizer"]["vocab_size"],
        d_model          = m["d_model"],
        n_heads          = m["n_heads"],
        n_encoder_layers = m["n_encoder_layers"],
        n_decoder_layers = m["n_decoder_layers"],
        d_ff             = m["d_ff"],
        max_len          = m["max_seq_len"],
        dropout          = m["dropout"],
        pad_id           = pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = LabelSmoothedCrossEntropy(cfg["tokenizer"]["vocab_size"],
                                          cfg["training"]["label_smoothing"], pad_id)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = InverseSqrtScheduler(optimizer, m["d_model"], cfg["training"]["warmup_steps"])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"]["fp16"] and device.type == "cuda")

    # Training loop
    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, cfg["tokenizer"]["vocab_size"]),
                                 tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            n_tokens = (tgt_out != pad_id).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

        train_loss = total_loss / total_tokens

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_in  = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                with torch.cuda.amp.autocast(enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                    logits = model(src, tgt_in)
                    loss = criterion(logits.reshape(-1, cfg["tokenizer"]["vocab_size"]),
                                     tgt_out.reshape(-1))
                n_tokens = (tgt_out != pad_id).sum().item()
                val_loss   += loss.item() * n_tokens
                val_tokens += n_tokens

        val_loss /= val_tokens
        print(f"Epoch {epoch:2d} | step {step:5d} | lr {scheduler.get_lr():.6f} "
              f"| train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, "checkpoints/best.pt")
            print(f"  → saved checkpoint (val loss {val_loss:.4f})")

        if step >= cfg["training"]["max_steps"]:
            print("Reached max steps.")
            break


if __name__ == "__main__":
    train("configs/thin_slice.yaml")