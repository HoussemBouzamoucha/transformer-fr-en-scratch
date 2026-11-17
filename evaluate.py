"""Translate a split with a trained checkpoint and score it (BLEU / chrF).

    python evaluate.py --config configs/thin_slice.yaml
    python evaluate.py --config configs/wmt14.yaml --split test --checkpoint checkpoints/wmt14/best.pt

Refuses to run if the tokenizer doesn't match the one the checkpoint was trained
with (different vocab file or threshold), because the ids would mean other tokens.
"""
import argparse
import sys
from pathlib import Path

import torch

from src.eval.metrics import evaluate_bleu
from src.model.transformer import Transformer
from src.tokenizer.loader import check_fingerprint, load_tokenizer
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thin_slice.yaml")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/<config name>/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--no-restrict", action="store_true", help="let the decoder emit French-only tokens")
    ap.add_argument("--force", action="store_true", help="skip the tokenizer/checkpoint check")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint or f"checkpoints/{Path(args.config).stem}/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not args.force:
        check_fingerprint(ckpt.get("tokenizer"), tokenizer)

    m = ckpt.get("config", cfg)["model"]
    model = Transformer(
        vocab_size=tokenizer.vocab_size, d_model=m["d_model"], n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"], n_decoder_layers=m["n_decoder_layers"],
        d_ff=m["d_ff"], max_len=m["max_seq_len"], dropout=0.0, pad_id=tokenizer.pad_id,
        tie_embeddings=m.get("shared_vocab", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    data_dir = Path(cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}"))
    src = [l.strip() for l in open(data_dir / f"{args.split}.fr", encoding="utf-8")]
    ref = [l.strip() for l in open(data_dir / f"{args.split}.en", encoding="utf-8")]

    res = evaluate_bleu(model, tokenizer, src, ref, device, batch_size=args.batch_size,
                        max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"],
                        restrict_output=not args.no_restrict)

    out = ckpt_path.parent / f"{args.split}.hyp.en"
    out.write_text("\n".join(res["hypotheses"]) + "\n", encoding="utf-8")
    print(f"{ckpt_path} (step {ckpt.get('step')}) on {data_dir}/{args.split}: "
          f"BLEU {res['bleu']:.2f} | chrF {res['chrf']:.2f}")
    print(f"  {res['bleu_signature']}")
    print(f"  translations -> {out}")
    for s, h, r in list(zip(src, res["hypotheses"], ref))[: args.examples]:
        print(f"\nFR  {s}\nOUT {h}\nREF {r}")


if __name__ == "__main__":
    main()
