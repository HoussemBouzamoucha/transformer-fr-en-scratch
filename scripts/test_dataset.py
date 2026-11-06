from torch.utils.data import DataLoader
from functools import partial
import sentencepiece as spm
from src.data.dataset import Multi30KDataset, collate_fn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SP_MODEL = "data/vocab/spm_8k.model"
DATA_DIR = "data/raw/multi30k"

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)
pad_id = sp.piece_to_id("<pad>")

dataset = Multi30KDataset("train", DATA_DIR, SP_MODEL)
print(f"Train pairs after length filter: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_id=pad_id)
)

src, tgt = next(iter(loader))
print(f"src shape: {src.shape}")
print(f"tgt shape: {tgt.shape}")
print(f"src[0]: {src[0].tolist()}")
print(f"tgt[0]: {tgt[0].tolist()}")