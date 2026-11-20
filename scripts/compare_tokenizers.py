"""Compare our joint BPE with pretrained tokenizers on the same FR/EN sentences.

    pip install transformers sentencepiece protobuf
    python scripts/compare_tokenizers.py                       # Multi30k test
    python scripts/compare_tokenizers.py --split val --md      # Markdown table too

Fair comparison rules:
  - same sentences for every tokenizer (raw lines of <data-dir>/<split>.fr/.en)
  - no special tokens counted (no <s>, </s>, language tags): add_special_tokens=False
  - our BPE is measured exactly as training encodes it (lang + vocab_threshold from the config)
Vocab size is printed next to every row: more tokens per sentence with a much
smaller vocab is expected, so compare at similar sizes or read both columns.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.loader import load_tokenizer
from src.utils import load_config

HF = {
    "GPT-2": "gpt2",
    "XLM-R": "xlm-roberta-base",
    "M2M-100": "facebook/m2m100_418M",
    "NLLB": "facebook/nllb-200-distilled-600M",
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/thin_slice.yaml")
ap.add_argument("--data-dir", default="data/raw/multi30k")
ap.add_argument("--split", default="test")
ap.add_argument("--only", nargs="*", help="subset of: ours " + " ".join(HF))
ap.add_argument("--md", action="store_true", help="also print a Markdown table")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

data = Path(args.data_dir)
lines = {lang: [l.strip() for l in open(data / f"{args.split}.{lang}", encoding="utf-8")]
         for lang in ("fr", "en")}
n_sent = len(lines["fr"])
words = {lang: sum(len(l.split()) for l in ls) for lang, ls in lines.items()}
chars = {lang: sum(len(l) for l in ls) for lang, ls in lines.items()}


def measure(name, vocab_size, encode, unk_id):
    row = {"name": name, "vocab": vocab_size}
    for lang, ls in lines.items():
        ids = [encode(l, lang) for l in ls]
        n = sum(len(x) for x in ids)
        row[lang] = {
            "per_sent": n / n_sent,
            "per_word": n / words[lang],          # fertility
            "chars_per_tok": chars[lang] / n,
            "unk": sum(x.count(unk_id) for x in ids) if unk_id is not None else 0,
        }
    return row


rows = []
want = set(args.only) if args.only else None

if want is None or "ours" in want:
    cfg = load_config(args.config)
    tok = load_tokenizer(cfg)
    label = f"Our BPE {tok.vocab_size // 1000}K"
    unk = tok.bpe.unk_id if hasattr(tok, "bpe") else None
    rows.append(measure(label, tok.vocab_size, lambda s, lang: tok.encode(s, lang=lang), unk))

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    print("transformers not installed: pip install transformers sentencepiece protobuf")

for name, repo in HF.items():
    if AutoTokenizer is None or (want is not None and name not in want):
        continue
    try:
        hf = AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # no internet, gated model, missing sentencepiece...
        print(f"skip {name} ({repo}): {type(e).__name__}: {e}")
        continue
    rows.append(measure(name, len(hf),
                        lambda s, lang, hf=hf: hf.encode(s, add_special_tokens=False),
                        hf.unk_token_id))

print(f"\n{n_sent} sentence pairs from {data}/{args.split}; "
      f"{words['fr'] / n_sent:.1f} FR / {words['en'] / n_sent:.1f} EN words per sentence\n")
hdr = (f"{'Tokenizer':<16}{'vocab':>9}  {'FR tok/sent':>11}{'EN tok/sent':>12}"
       f"{'FR tok/word':>12}{'EN tok/word':>12}{'FR ch/tok':>10}{'EN ch/tok':>10}{'<unk>':>7}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:<16}{r['vocab']:>9,}  {r['fr']['per_sent']:>11.1f}{r['en']['per_sent']:>12.1f}"
          f"{r['fr']['per_word']:>12.2f}{r['en']['per_word']:>12.2f}"
          f"{r['fr']['chars_per_tok']:>10.2f}{r['en']['chars_per_tok']:>10.2f}"
          f"{r['fr']['unk'] + r['en']['unk']:>7}")

if args.md:
    print("\n| Tokenizer | Vocab | FR tokens/sentence | EN tokens/sentence | FR tokens/word | EN tokens/word | <unk> |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['vocab']:,} | {r['fr']['per_sent']:.1f} | {r['en']['per_sent']:.1f} | "
              f"{r['fr']['per_word']:.2f} | {r['en']['per_word']:.2f} | {r['fr']['unk'] + r['en']['unk']} |")
