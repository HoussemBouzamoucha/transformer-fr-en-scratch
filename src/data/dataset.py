import torch
from torch.utils.data import Dataset
from pathlib import Path


class Multi30KDataset(Dataset):
    def __init__(self, split: str, data_dir: str, tokenizer,
                 max_len: int = 128, max_pairs: int = None):
        """tokenizer: anything from src.tokenizer.loader.load_tokenizer.
        It is only used here, not stored, so DataLoader workers don't pickle it."""
        data_dir = Path(data_dir)
        self.max_len = max_len

        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        fr_path = data_dir / f"{split}.fr"
        en_path = data_dir / f"{split}.en"

        with open(fr_path, encoding="utf-8") as f:
            fr_lines = [l.strip() for l in f]
        with open(en_path, encoding="utf-8") as f:
            en_lines = [l.strip() for l in f]

        assert len(fr_lines) == len(en_lines), "FR and EN files must have same number of lines"

        self.pairs = []
        skipped = 0
        for fr, en in zip(fr_lines, en_lines):
            if max_pairs and len(self.pairs) >= max_pairs:
                break

            fr_ids = [self.bos_id] + tokenizer.encode(fr, lang="fr") + [self.eos_id]
            en_ids = [self.bos_id] + tokenizer.encode(en, lang="en") + [self.eos_id]

            if len(fr_ids) > max_len or len(en_ids) > max_len:
                skipped += 1
                continue

            self.pairs.append((fr_ids, en_ids))

        print(f"[{split}] {len(self.pairs)} pairs loaded, {skipped} skipped (too long)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fr_ids, en_ids = self.pairs[idx]
        return torch.tensor(fr_ids, dtype=torch.long), torch.tensor(en_ids, dtype=torch.long)


def collate_fn(batch, pad_id: int):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded