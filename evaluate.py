"""Translate a split with a trained checkpoint and score it (BLEU / chrF).

    python evaluate.py --config configs/thin_slice.yaml
    python evaluate.py --config configs/wmt14.yaml --split test --checkpoint checkpoints/wmt14/best.pt

Refuses to run if the tokenizer doesn't match the one the checkpoint was trained
with (different vocab file or threshold), because the ids would mean other tokens.
"""
import argparse
import sys
from pathlib import Path

from src.eval.metrics import evaluate_bleu, decoding_options
from src.inference.translator import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thin_slice.yaml")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/<config name>/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--no-restrict", action="store_true", help="let the decoder emit French-only tokens")
    ap.add_argument("--force", action="store_true", help="skip the tokenizer/checkpoint check")
    ap.add_argument("--beam", type=int, default=None, help="beam size (default: inference.beam_size, 1 = greedy)")
    ap.add_argument("--alpha", type=float, default=None, help="length penalty (default: inference.length_penalty)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    model, tokenizer, cfg, ckpt, ckpt_path, device = load_model(
        args.config, args.checkpoint, force=args.force)
    m = ckpt.get("config", cfg)["model"]
    dec = decoding_options(cfg)
    if args.beam is not None:
        dec["beam_size"] = args.beam
    if args.alpha is not None:
        dec["length_penalty"] = args.alpha

    data_dir = Path(cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}"))
    src = [l.strip() for l in open(data_dir / f"{args.split}.fr", encoding="utf-8")]
    ref = [l.strip() for l in open(data_dir / f"{args.split}.en", encoding="utf-8")]

    res = evaluate_bleu(model, tokenizer, src, ref, device, batch_size=args.batch_size,
                        max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"],
                        restrict_output=not args.no_restrict, **dec)

    out = ckpt_path.parent / f"{args.split}.hyp.en"
    out.write_text("\n".join(res["hypotheses"]) + "\n", encoding="utf-8")
    how = (f"beam {dec['beam_size']}, alpha {dec['length_penalty']}" if dec["beam_size"] > 1 else "greedy")
    if ckpt.get("averaged_from"):
        how += f", average of {len(ckpt['averaged_from'])} checkpoints"
    print(f"{ckpt_path} (step {ckpt.get('step')}, {how}) on {data_dir}/{args.split}: "
          f"BLEU {res['bleu']:.2f} | chrF {res['chrf']:.2f}")
    print(f"  {res['bleu_signature']}")
    print(f"  translations -> {out}")
    for s, h, r in list(zip(src, res["hypotheses"], ref))[: args.examples]:
        print(f"\nFR  {s}\nOUT {h}\nREF {r}")


if __name__ == "__main__":
    main()
