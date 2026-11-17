"""Inspect the joint BPE: first / random / last 100 vocab entries and test sentences.

    python scripts/inspect_bpe.py
    python scripts/inspect_bpe.py --bpe data/vocab/own_bpe_8k_wmt.json --data-dir data/raw/wmt14 --split test

Prints everything and writes the same tables to --out-dir:
    vocab_first100.txt  vocab_random100.txt  vocab_last100.txt   (token<TAB>id per line)
    sents_first100.tsv  sents_random100.tsv  sents_last100.tsv
"""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--bpe", default="data/vocab/own_bpe_8k_multi30k.json")
parser.add_argument("--data-dir", default="data/raw/multi30k")
parser.add_argument("--split", default="test")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--threshold", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default="logs/bpe_inspect")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")  # Windows console + accents
rng = random.Random(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
bpe = BPETokenizer().load(args.bpe)


def pick(items, n):
    """first n, random n (kept in original order), last n — as (index, item) lists."""
    idx = list(range(len(items)))
    rand = sorted(rng.sample(idx, min(n, len(idx))))
    return {
        "first100": [(i, items[i]) for i in idx[:n]],
        "random100": [(i, items[i]) for i in rand],
        "last100": [(i, items[i]) for i in idx[-n:]],
    }


def emit(name, header, rows):
    path = out_dir / f"{name}.tsv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n=== {name} ({len(rows)} rows) -> {path}")
    for r in rows:
        print("  " + " | ".join(str(x) for x in r))


# ---- vocab: token and id
vocab = sorted(bpe.vocab.items(), key=lambda kv: kv[1])  # by id
for name, picked in pick(vocab, args.n).items():
    # One "token<TAB>id" per line. Tokens never contain whitespace, so this splits cleanly.
    path = out_dir / f"vocab_{name}.txt"
    lines = [f"{tok}\t{tid}" for _, (tok, tid) in picked]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n=== vocab_{name} ({len(lines)} tokens) -> {path}")
    print("\n".join(lines))

# ---- sentences: tokenized FR and EN side by side
data_dir = Path(args.data_dir)
fr = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.fr", encoding="utf-8")]
en = [l.rstrip("\n") for l in open(data_dir / f"{args.split}.en", encoding="utf-8")]
assert len(fr) == len(en), "FR and EN files must have the same number of lines"

stats = {"fr": [0, 0], "en": [0, 0]}  # [tokens, unk]
for name, picked in pick(list(zip(fr, en)), args.n).items():
    rows = []
    for i, (f_txt, e_txt) in picked:
        f_ids = bpe.encode(f_txt, lang="fr", threshold=args.threshold)
        e_ids = bpe.encode(e_txt, lang="en", threshold=args.threshold)
        for lang, ids in (("fr", f_ids), ("en", e_ids)):
            stats[lang][0] += len(ids)
            stats[lang][1] += ids.count(bpe.unk_id)
        rows.append((i, len(f_ids), " ".join(bpe.id_to_token[t] for t in f_ids),
                     len(e_ids), " ".join(bpe.id_to_token[t] for t in e_ids)))
    emit(f"sents_{name}", ["line", "fr_len", "fr_tokens", "en_len", "en_tokens"], rows)

n_sents = 3 * args.n
print(f"\nSummary over {n_sents} sampled sentence pairs (threshold={args.threshold}):")
for lang, (toks, unk) in stats.items():
    print(f"  {lang}: {toks / n_sents:.1f} tokens/sentence, {unk} <unk>")
print(f"Files written to {out_dir.resolve()}")
