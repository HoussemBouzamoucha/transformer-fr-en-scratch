import re
from collections import defaultdict
from heapq import heappush, heappop


class BPETokenizer:
    def __init__(self):
        self.merges = {}        # (a, b) -> merged token
        self.vocab = {}         # token -> id
        self.id_to_token = {}   # id -> token
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

    def _get_word_freqs(self, texts):
        """Split text into characters with end-of-word marker, count frequencies."""
        word_freqs = defaultdict(int)
        for text in texts:
            for word in text.strip().split():
                # Represent word as tuple of chars + end marker
                chars = tuple(list(word) + ["</w>"])
                word_freqs[chars] += 1
        return word_freqs

    def _get_pair_counts(self, word_freqs):
        """Count frequency of each adjacent pair across all words."""
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        return pair_counts

    def _merge_pair(self, pair, word_freqs):
        """Merge all occurrences of pair in word_freqs."""
        new_word_freqs = {}
        a, b = pair
        merged = a + b
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, texts, vocab_size: int, verbose: bool = False):
        """Train BPE on a list of strings."""
        # Start with character vocab
        word_freqs = self._get_word_freqs(texts)

        # Base vocab: all unique characters + special tokens
        base_vocab = set()
        for word in word_freqs:
            for char in word:
                base_vocab.add(char)

        # Assign ids: special tokens first
        self.vocab = {}
        for tok in self.special_tokens:
            self.vocab[tok] = len(self.vocab)
        for char in sorted(base_vocab):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # BPE merges
        num_merges = vocab_size - len(self.vocab)
        self.merges = {}

        for i in range(num_merges):
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break

            # Pick most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break

            # Merge
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            self.vocab[merged] = len(self.vocab)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            if verbose and i % 500 == 0:
                print(f"  Merge {i}: {best_pair} -> {merged} (count={best_count})")

        # Build id_to_token
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Training done. Vocab size: {len(self.vocab)}")

    def _tokenize_word(self, word):
        """Apply learned merges to a single word."""
        chars = list(word) + ["</w>"]
        while True:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            # Find the earliest merge in our merge list
            mergeable = [(self.merges.index((p[0], p[1])) if (p[0], p[1]) in self.merges else float("inf"), i, p)
                         for i, p in enumerate(pairs)]
            best = min(mergeable, key=lambda x: x[0])
            if best[0] == float("inf"):
                break
            _, idx, (a, b) = best
            merged = a + b
            chars = chars[:idx] + [merged] + chars[idx + 2:]
        return chars

    def encode(self, text: str):
        """Encode text to token ids."""
        bos = self.vocab.get("<s>", 2)
        eos = self.vocab.get("</s>", 3)
        unk = self.vocab.get("<unk>", 1)

        ids = [bos]
        for word in text.strip().split():
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self.vocab.get(tok, unk))
        ids.append(eos)
        return ids

    def decode(self, ids):
        """Decode token ids back to text."""
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids
                  if i not in {self.vocab.get("<s>"), self.vocab.get("</s>"),
                                self.vocab.get("<pad>")}]
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        return text

    def save(self, path: str):
        import json
        data = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {path}")

    def load(self, path: str):
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {(k[0], k[1]): v for k, v in data["merges"]}
        print(f"Loaded vocab size: {len(self.vocab)}")