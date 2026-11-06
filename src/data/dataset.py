# token-based batching, bucketing   import torch
from tensorboard import lazy
from torch.utils.data import Dataset
from pathlib import Path
import sentencepiece as spm
import torch


class Multi30KDataset(Dataset):
    def __init__(self, split: str, data_dir: str, sp_model_path: str, max_len: int = 128):
        """
        split: 'train', 'val', or 'test'
        """
        data_dir = Path(data_dir)
        self.max_len = max_len

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.piece_to_id("<s>")
        self.eos_id = self.sp.piece_to_id("</s>")

        # Load raw text
        fr_path = data_dir / f"{split}.fr"
        en_path = data_dir / f"{split}.en"

        with open(fr_path, encoding="utf-8") as f:
            fr_lines = [l.strip() for l in f]
        with open(en_path, encoding="utf-8") as f:
            en_lines = [l.strip() for l in f]

        assert len(fr_lines) == len(en_lines), "FR and EN files must have same number of lines"

        # Tokenize everything upfront (fast on Multi30K)
        self.pairs = []
        for fr, en in zip(fr_lines, en_lines):
            fr_ids = [self.bos_id] + self.sp.encode(fr) + [self.eos_id]
            en_ids = [self.bos_id] + self.sp.encode(en) + [self.eos_id]

            # Length filter
            if len(fr_ids) > max_len or len(en_ids) > max_len:
                continue

            self.pairs.append((fr_ids, en_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fr_ids, en_ids = self.pairs[idx]
        return torch.tensor(fr_ids, dtype=torch.long), torch.tensor(en_ids, dtype=torch.long)


def collate_fn(batch, pad_id: int):
    """Pads a batch of (src, tgt) pairs to the same length."""
    src_batch, tgt_batch = zip(*batch)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    return src_padded, tgt_padded