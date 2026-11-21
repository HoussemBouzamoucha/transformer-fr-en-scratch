from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
        # Ids are kept as flat int32 arrays + offsets, not lists of Python ints:
        # ~0.3 GB instead of several GB for 1M pairs, and DataLoader workers
        # share the arrays instead of slowly copying them.
        src_parts, tgt_parts, src_lens, tgt_lens = [], [], [], []
        skipped = 0
        for start in range(0, len(fr_lines), chunk):
            if max_pairs and len(src_lens) >= max_pairs:
                break
            fr_batch = encode_batch(fr_lines[start:start + chunk], lang="fr")
            en_batch = encode_batch(en_lines[start:start + chunk], lang="en")
            flat_s, flat_t = [], []
            for fr, en in zip(fr_batch, en_batch):
                if max_pairs and len(src_lens) >= max_pairs:
                    break
                if len(fr) + 2 > max_len or len(en) + 2 > max_len:  # + <s> </s>
                    skipped += 1
                    continue
                flat_s.append(self.bos_id); flat_s.extend(fr); flat_s.append(self.eos_id)
                flat_t.append(self.bos_id); flat_t.extend(en); flat_t.append(self.eos_id)
                src_lens.append(len(fr) + 2)
                tgt_lens.append(len(en) + 2)
            src_parts.append(np.array(flat_s, dtype=np.int32))
            tgt_parts.append(np.array(flat_t, dtype=np.int32))

        self._src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int32)
        self._tgt = np.concatenate(tgt_parts) if tgt_parts else np.zeros(0, dtype=np.int32)
        self._src_off = np.zeros(len(src_lens) + 1, dtype=np.int64)
        self._tgt_off = np.zeros(len(tgt_lens) + 1, dtype=np.int64)
        np.cumsum(src_lens, out=self._src_off[1:])
        np.cumsum(tgt_lens, out=self._tgt_off[1:])
        self.lengths = list(zip(src_lens, tgt_lens))  # (src, tgt) with <s> </s>, per pair

        print(f"[{split}] {len(self.lengths)} pairs loaded, {skipped} skipped (too long)")

    @property
    def pairs(self):
        """(fr ids, en ids) lists per pair, built on demand (for inspection and tests)."""
        return [self._pair_lists(i) for i in range(len(self))]

    def _pair_lists(self, i):
        return (self._src[self._src_off[i]:self._src_off[i + 1]].tolist(),
                self._tgt[self._tgt_off[i]:self._tgt_off[i + 1]].tolist())

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        src = self._src[self._src_off[idx]:self._src_off[idx + 1]]
        tgt = self._tgt[self._tgt_off[idx]:self._tgt_off[idx + 1]]
        return torch.from_numpy(src.astype(np.int64)), torch.from_numpy(tgt.astype(np.int64))


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
        self.skip = 0  # batches to skip at the start of the next epoch (resuming mid-epoch)
        longest = max(max(s, t) for s, t in lengths)
        if longest > max_tokens:
            raise ValueError(f"max_tokens={max_tokens} is smaller than the longest pair ({longest})")
        self._n_batches = len(self._batches())

    @classmethod
    def from_dataset(cls, dataset, max_tokens, shuffle=True, seed=0):
        lengths = getattr(dataset, "lengths", None) or [(len(s), len(t)) for s, t in dataset.pairs]
        return cls(lengths, max_tokens, shuffle, seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.skip = 0  # set it again after this call to resume mid-epoch

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
        # The batch order is a pure function of (seed, epoch), so a resumed run
        # skips exactly the batches it already trained on in this epoch. skip is
        # not reset here: a multi-worker DataLoader calls __iter__ twice per epoch.
        return iter(self._batches()[self.skip:])

    def __len__(self):
        return self._n_batches
