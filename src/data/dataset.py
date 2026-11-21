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

        # Encoded in big batches (multithreaded C++ for our BPE); same ids as encode().
        encode_batch = getattr(tokenizer, "encode_batch", None) or (
            lambda texts, lang: [tokenizer.encode(t, lang=lang) for t in texts])
        chunk = min(200_000, max_pairs) if max_pairs else 200_000
        self.pairs = []
        skipped = 0
        for start in range(0, len(fr_lines), chunk):
            if max_pairs and len(self.pairs) >= max_pairs:
                break
            fr_batch = encode_batch(fr_lines[start:start + chunk], lang="fr")
            en_batch = encode_batch(en_lines[start:start + chunk], lang="en")
            for fr, en in zip(fr_batch, en_batch):
                if max_pairs and len(self.pairs) >= max_pairs:
                    break

                fr_ids = [self.bos_id] + fr + [self.eos_id]
                en_ids = [self.bos_id] + en + [self.eos_id]

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

class TokenBatchSampler(torch.utils.data.Sampler):
    """Batches of sentence pairs with similar length, ~max_tokens tokens each.

    Vaswani et al. 2017 sec. 5.1: "Sentence pairs were batched together by
    approximate sequence length. Each training batch contained a set of sentence
    pairs containing approximately 25000 source tokens and 25000 target tokens."
    A batch is closed when its padded source or target size would exceed
    max_tokens; sorting by length keeps that padding small. Pairs of equal length
    are shuffled and the batch order is shuffled every epoch (call set_epoch).
    """
    def __init__(self, lengths, max_tokens: int, shuffle: bool = True, seed: int = 0):
        """lengths: list of (src_len, tgt_len) per pair, in dataset order."""
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        longest = max(max(s, t) for s, t in lengths)
        if longest > max_tokens:
            raise ValueError(f"max_tokens={max_tokens} is smaller than the longest pair ({longest})")
        self._n_batches = len(self._batches())

    @classmethod
    def from_dataset(cls, dataset, max_tokens, shuffle=True, seed=0):
        return cls([(len(s), len(t)) for s, t in dataset.pairs], max_tokens, shuffle, seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist() if self.shuffle \
            else list(range(len(self.lengths)))
        # Stable sort: equal-length pairs keep the random order from randperm.
        order.sort(key=lambda i: (self.lengths[i][1], self.lengths[i][0]))

        batches, cur, max_s, max_t = [], [], 0, 0
        for i in order:
            s, t = self.lengths[i]
            new_s, new_t = max(max_s, s), max(max_t, t)
            if cur and (len(cur) + 1) * max(new_s, new_t) > self.max_tokens:
                batches.append(cur)
                cur, new_s, new_t = [], s, t
            cur.append(i)
            max_s, max_t = new_s, new_t
        if cur:
            batches.append(cur)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[j] for j in perm]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return self._n_batches
