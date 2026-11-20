"""Clean and sample a parallel FR/EN training set (run on Colab for WMT14).

    python scripts/clean_data.py --src-dir data/raw/wmt14 --out-dir data/clean/wmt14 --max-pairs 200000

train:   random sample of pairs from the WHOLE corpus (WMT14 concatenates several
         corpora, so the first N lines are all one domain), then normalize, filter,
         de-duplicate. The BPE and the model then both use this same set.
val/test: copied unchanged. Never filter evaluation data, and BLEU must be
         scored against the original references (the tokenizer normalizes the
         source side itself at encode time).

No language-ID filter: it would need an extra dependency (e.g. fasttext lid.176).
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.normalize import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--src-dir", default="data/raw/wmt14")
ap.add_argument("--out-dir", default="data/clean/wmt14")
ap.add_argument("--max-pairs", type=int, default=None, help="random sample size (default: keep all)")
ap.add_argument("--oversample", type=float, default=1.3,
                help="sample this many times --max-pairs before filtering, so enough survive")
ap.add_argument("--max-words", type=int, default=150)
ap.add_argument("--max-ratio", type=float, default=2.5, help="max char-length ratio between sides")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

src, out = Path(args.src_dir), Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(args.seed)


def pairs():
    with open(src / "train.fr", encoding="utf-8") as f, open(src / "train.en", encoding="utf-8") as e:
        for i, (fr, en) in enumerate(zip(f, e)):
            if i and i % 2_000_000 == 0:
                print(f"  read {i:,} lines")
            yield fr, en


# 1) Reservoir sample over the whole file (one pass, O(sample) memory).
if args.max_pairs:
    k = int(args.max_pairs * args.oversample)
    sample, n = [], 0
    for n, p in enumerate(pairs(), 1):
        if len(sample) < k:
            sample.append(p)
        else:
            j = rng.randrange(n)
            if j < k:
                sample[j] = p
    rng.shuffle(sample)
    print(f"Read {n:,} pairs, sampled {len(sample):,}")
else:
    sample = list(pairs())
    print(f"Read {len(sample):,} pairs")

# 2) Normalize + filter + dedup.
reasons = {"empty": 0, "too_long": 0, "ratio": 0, "copy": 0, "duplicate": 0}
seen, kept = set(), []
for fr, en in sample:
    fr, en = normalize(fr), normalize(en)
    if not fr or not en:
        reasons["empty"] += 1
        continue
    if len(fr.split()) > args.max_words or len(en.split()) > args.max_words:
        reasons["too_long"] += 1
        continue
    a, b = len(fr), len(en)
    if max(a, b) > 10 and max(a, b) / max(min(a, b), 1) > args.max_ratio:
        reasons["ratio"] += 1
        continue
    if fr == en:  # untranslated copy
        reasons["copy"] += 1
        continue
    key = hash((fr, en))
    if key in seen:
        reasons["duplicate"] += 1
        continue
    seen.add(key)
    kept.append((fr, en))
    if args.max_pairs and len(kept) >= args.max_pairs:
        break

if args.max_pairs and len(kept) < args.max_pairs:
    print(f"WARNING: only {len(kept):,} pairs survived; raise --oversample")

with open(out / "train.fr", "w", encoding="utf-8") as f, open(out / "train.en", "w", encoding="utf-8") as e:
    for fr, en in kept:
        f.write(fr + "\n")
        e.write(en + "\n")
print(f"Kept {len(kept):,} train pairs -> {out}; dropped: {reasons}")

for split in ("val", "test"):
    for lang in ("fr", "en"):
        p = src / f"{split}.{lang}"
        if p.exists():
            shutil.copy(p, out / p.name)
            print(f"  copied {p} unchanged")
        else:
            print(f"  missing {p} (dataset/eval expect it)")
