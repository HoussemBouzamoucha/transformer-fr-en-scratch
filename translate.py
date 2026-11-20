"""Translate your own French sentences with a trained checkpoint.

    python translate.py "Un chat noir dort sur le canapé." "Deux enfants jouent au football."
    python translate.py --file mes_phrases.txt          # one sentence per line
    python translate.py                                 # type sentences, empty line to quit
    python translate.py --tokens "Un chat noir dort."   # also show the BPE tokens

Options: --config (default configs/thin_slice.yaml), --checkpoint (default
checkpoints/<config name>/best.pt), --no-restrict, --force.
"""
import argparse
import sys

from src.inference.translator import Translator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--config", default="configs/thin_slice.yaml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokens", action="store_true", help="show BPE tokens too")
    ap.add_argument("--no-restrict", action="store_true", help="let the decoder emit French-only tokens")
    ap.add_argument("--force", action="store_true", help="skip the tokenizer/checkpoint check")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tr = Translator(args.config, args.checkpoint, restrict_output=not args.no_restrict, force=args.force)
    show = tr.explain if args.tokens else (lambda s: print(f"FR  {s}\nEN  {tr(s)}"))

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if args.tokens:
            for s in lines:
                show(s)
                print()
        else:
            for s, t in zip(lines, tr(lines)):  # batched
                print(f"FR  {s}\nEN  {t}\n")
    elif args.sentences:
        for s in args.sentences:
            show(s)
            print()
    else:
        print("Type a French sentence (empty line to quit).")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                break
            if not s:
                break
            show(s)


if __name__ == "__main__":
    main()
