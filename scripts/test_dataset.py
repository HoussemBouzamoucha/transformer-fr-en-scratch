from torch.utils.data import DataLoader
from functools import partial
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import Multi30KDataset, collate_fn
from src.tokenizer.loader import load_tokenizer
from src.utils import load_config


CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/thin_slice.yaml"
DATA_DIR = "data/raw/multi30k"

tokenizer = load_tokenizer(load_config(CONFIG))
pad_id = tokenizer.pad_id

dataset = Multi30KDataset("train", DATA_DIR, tokenizer)
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