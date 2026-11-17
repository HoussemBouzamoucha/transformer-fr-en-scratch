"""Train the joint FR/EN BPE tokenizer.

    python scripts/train_own_bpe.py --data-dir data/raw/multi30k --out data/vocab/own_bpe_8k_multi30k.json
    python scripts/train_own_bpe.py --data-dir data/raw/wmt14 --out data/vocab/own_bpe_8k_wmt.json
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/raw/multi30k")
parser.add_argument("--out", default="data/vocab/own_bpe_8k_multi30k.json")
parser.add_argument("--vocab-size", type=int, default=8000)
parser.add_argument("--sample-size", type=int, default=100_000,
                    help="max lines per language (same count for both keeps the vocab balanced)")
parser.add_argument("--threshold", type=int, default=50,
                    help="vocab threshold used only for the sanity check printout")
args = parser.parse_args()

data_dir = Path(args.data_dir)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

corpora = {}
for lang in ["fr", "en"]:
    with open(data_dir / f"train.{lang}", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= args.sample_size:
                break
            lines.append(line.strip())
    corpora[lang] = lines
    print(f"  train.{lang}: {len(lines)} lines")

print(f"\nTraining joint BPE (vocab_size={args.vocab_size})...")
t0 = time.time()
bpe = BPETokenizer()
bpe.train(corpora, vocab_size=args.vocab_size, verbose=True)
print(f"Took {time.time() - t0:.1f}s")
bpe.save(str(out_path))

# How much of the joint vocab each language actually uses.
for lang, counts in bpe.lang_counts.items():
    kept = sum(c >= args.threshold for c in counts.values())
    print(f"  {lang}: uses {len(counts)} tokens, {kept} seen >= {args.threshold} times")

print("\nSanity check:")
samples = {"fr": "Un homme regarde une fenêtre.", "en": "A man is looking at a window."}
for lang, text in samples.items():
    plain = bpe.tokenize(text)
    filtered = bpe.tokenize(text, lang=lang, threshold=args.threshold)
    print(f"  [{lang}] {text}")
    print(f"       joint:    {plain}")
    print(f"       filtered: {filtered}")
    assert bpe.decode(bpe.encode(text)) == text
