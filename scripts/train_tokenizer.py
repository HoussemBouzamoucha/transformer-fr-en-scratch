import sentencepiece as spm
from pathlib import Path

RAW_DIR = Path("data/raw/multi30k")
VOCAB_DIR = Path("data/vocab")
VOCAB_DIR.mkdir(parents=True, exist_ok=True)

# Combine train.fr and train.en into one file for shared vocab
combined_path = VOCAB_DIR / "train_combined.txt"

print("Combining train files...")
with open(combined_path, "w", encoding="utf-8") as out:
    for fname in ["train.fr", "train.en"]:
        with open(RAW_DIR / fname, encoding="utf-8") as f:
            for line in f:
                out.write(line)

total = sum(1 for _ in open(combined_path, encoding="utf-8"))
print(f"Combined file: {total} lines")

# Train SentencePiece BPE
print("\nTraining SentencePiece (BPE, vocab_size=8000)...")
spm.SentencePieceTrainer.train(
    input=str(combined_path),
    model_prefix=str(VOCAB_DIR / "spm_8k"),
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
)

print("Done. Files saved:")
for f in sorted(VOCAB_DIR.iterdir()):
    print(f"  {f}")

# Quick sanity check
print("\nSanity check:")
sp = spm.SentencePieceProcessor()
sp.load(str(VOCAB_DIR / "spm_8k.model"))

test_fr = "Un homme regarde une fenêtre."
test_en = "A man is looking at a window."

print(f"  FR: {test_fr}")
print(f"      {sp.encode(test_fr, out_type=str)}")
print(f"  EN: {test_en}")
print(f"      {sp.encode(test_en, out_type=str)}")