import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

DATA_DIR = Path("data/raw/wmt14")
VOCAB_DIR = Path("data/vocab")
VOCAB_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 100000  # lines per language
VOCAB_SIZE  = 8000

print("Loading training data sample...")
texts = []
for fname in ["train.fr", "train.en"]:
    with open(DATA_DIR / fname, encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            lines.append(line.strip())
    texts.extend(lines)
    print(f"  {fname}: {len(lines)} lines")

print(f"Total lines: {len(texts)}")

print(f"\nTraining BPE (vocab_size={VOCAB_SIZE})...")
bpe = BPETokenizer()
bpe.train(texts, vocab_size=VOCAB_SIZE, verbose=True)

out_path = VOCAB_DIR / "own_bpe_8k.json"
bpe.save(str(out_path))
print(f"\nDone. Saved to {out_path}")

# Sanity check
print("\nSanity check:")
test_fr = "Un homme regarde une fenêtre."
test_en = "A man is looking at a window."
for text in [test_fr, test_en]:
    ids = bpe.encode(text)
    tokens = [bpe.id_to_token[i] for i in ids[1:-1]]
    print(f"  {text}")
    print(f"  {tokens}")