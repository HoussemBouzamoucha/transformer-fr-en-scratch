"""Train the joint FR/EN BPE tokenizer.

    python scripts/train_own_bpe.py --data-dir data/raw/multi30k --out data/vocab/own_bpe_8k_multi30k.json
    python scripts/train_own_bpe.py --data-dir data/clean/wmt14  --out data/vocab/own_bpe_8k_wmt.json

Point --data-dir at the same training set the model uses (for WMT: the output of
scripts/clean_data.py). Lines are sampled at random over the whole file, with the
same line numbers on both sides, so the sample is balanced and covers every domain.
Afterwards it prints how much each vocab threshold lengthens validation sentences,
to choose tokenizer.vocab_threshold in the config.

The vocab is fitted to --vocab-threshold (default 0, as in thin_slice.yaml): every
slot goes to a token the training text really uses at that threshold. Use the
same value as tokenizer.vocab_threshold in the config, or slots are wasted.
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.sampling import sample_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/raw/multi30k")
parser.add_argument("--out", default="data/vocab/own_bpe_8k_multi30k.json")
parser.add_argument("--vocab-size", type=int, default=8000)
parser.add_argument("--sample-size", type=int, default=100_000,
                    help="max sentence pairs, sampled at random over the whole file")
parser.add_argument("--min-char-count", type=int, default=5,
                    help="rarer characters become <unk> (ASCII and French letters always kept)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--vocab-threshold", type=int, default=0,
                    help="the tokenizer.vocab_threshold the model will use; the vocab is filled "
                         "with tokens actually used at that threshold (-1 = classic BPE, no fitting)")
parser.add_argument("--no-pretokenize", action="store_true",
                    help="old behaviour: keep punctuation attached to words (for comparisons)")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

data_dir = Path(args.data_dir)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

corpora, n = sample_pairs(data_dir, args.sample_size, args.seed)
sample = corpora["fr"]
print(f"Sampled {len(sample):,} of {n:,} pairs from {data_dir}")

print(f"\nTraining joint BPE (vocab_size={args.vocab_size})...")
t0 = time.time()
bpe = BPETokenizer(pretokenize=0 if args.no_pretokenize else 2)
bpe.train(corpora, vocab_size=args.vocab_size, min_char_count=args.min_char_count,
          fit_threshold=None if args.vocab_threshold < 0 else args.vocab_threshold)
print(f"Took {time.time() - t0:.1f}s")
bpe.save(str(out_path))

# Threshold table on held-out text: how much longer sentences get.
held = {}
for lang in ("fr", "en"):
    p = data_dir / f"val.{lang}"
    held[lang] = ([l.strip() for l in open(p, encoding="utf-8")] if p.exists()
                  else corpora[lang][-2000:])[:3000]
print("\nVocab threshold -> extra tokens per sentence on validation (pick tokenizer.vocab_threshold):")
# encode_batch: the multithreaded C++ encoder (same ids as encode()).
base_ids = {lang: bpe.encode_batch(lines, lang=lang) for lang, lines in held.items()}
base = {lang: sum(map(len, ids)) for lang, ids in base_ids.items()}
print("  threshold    " + "    ".join(f"{lang}" for lang in held))
for t in (0, 1, 2, 5, 10, 20, 50):
    cells = []
    for lang, lines in held.items():
        n_tok = sum(map(len, bpe.encode_batch(lines, lang=lang, threshold=t)))
        cells.append(f"{100 * (n_tok / base[lang] - 1):+5.1f}%")
    print(f"  {t:>9}    " + "   ".join(cells))
unk = {lang: sum(x.count(bpe.unk_id) for x in ids) for lang, ids in base_ids.items()}
print(f"  <unk> on validation: {unk}")

print("\nSanity check:")
samples = {"fr": "Un homme regarde l’herbe, près d’une fenêtre.", "en": "A man's looking at the grass, near a window."}
for lang, text in samples.items():
    print(f"  [{lang}] {text}")
    print(f"       {bpe.tokenize(text, lang=lang)}")
    print(f"       decoded: {bpe.decode(bpe.encode(text, lang=lang))}")
