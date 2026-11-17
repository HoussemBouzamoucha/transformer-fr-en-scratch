"""Joint BPE tokenizer (Sennrich et al., 2016) for a shared FR/EN vocabulary.

One set of merges and one vocabulary are learned from both languages together.
Each language also keeps its own token counts. At encode time those counts can
undo merges that were (almost) only ever seen in the *other* language, which is
the "vocabulary threshold" trick from subword-nmt.
"""
import json
import re
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush

# Letters/digits stay together; every other non-space character is its own piece.
_PIECE_RE = re.compile(r"\w+|[^\w\s]")


class BPETokenizer:
    EOW = "</w>"
    # Join mark for pre-tokenization: on a punctuation piece, "⁐." means "no space
    # before me" and "'⁐" means "no space after me". Word pieces never carry it, so
    # "fenêtre" is the same token in "fenêtre." and "fenêtre ouverte".
    GLUE = "⁐"

    def __init__(self, pretokenize: bool = True):
        self.pretokenize = pretokenize
        self.merges = []          # ordered list of (a, b); index = merge rank
        self.merge_ranks = {}     # (a, b) -> rank
        self.split_of = {}        # merged token -> (a, b) that first produced it
        self.vocab = {}           # token -> id
        self.id_to_token = {}     # id -> token
        self.base_tokens = set()  # single characters + </w>; never split further
        self.lang_counts = {}     # lang -> {token: count in that language}
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self._cache = {}

    # ------------------------------------------------------------------ ids
    @property
    def pad_id(self):
        return self.vocab["<pad>"]

    @property
    def unk_id(self):
        return self.vocab["<unk>"]

    @property
    def bos_id(self):
        return self.vocab["<s>"]

    @property
    def eos_id(self):
        return self.vocab["</s>"]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def piece_to_id(self, piece):
        return self.vocab.get(piece, self.unk_id)

    # ------------------------------------------------------- pre-tokenizing
    def _pieces(self, word):
        """Split one whitespace-delimited word into BPE input pieces.

        Each boundary inside the word gets one GLUE mark, on the punctuation side:
        suffix of the left piece if it is punctuation, else prefix of the right one.
        """
        if not self.pretokenize:
            return [word]
        word = word.replace(self.GLUE, "")
        parts = _PIECE_RE.findall(word)
        if len(parts) <= 1:
            return parts or [word]
        out = list(parts)
        for i in range(len(parts) - 1):
            left_is_punct = not (parts[i][0].isalnum() or parts[i][0] == "_")
            if left_is_punct:
                out[i] = out[i] + self.GLUE
            else:
                out[i + 1] = self.GLUE + out[i + 1]
        return out

    def _join(self, text):
        """Inverse of _pieces over a whole decoded string ("a</w>⁐.</w>" -> "a.")."""
        pieces = [p for p in text.split(self.EOW)]
        if pieces and pieces[-1] == "":
            pieces.pop()
        out = []
        for i, p in enumerate(pieces):
            if i > 0 and not (pieces[i - 1].endswith(self.GLUE) or p.startswith(self.GLUE)):
                out.append(" ")
            out.append(p)
        return "".join(out).replace(self.GLUE, "").strip()

    # ------------------------------------------------------------- training
    def _word_counts(self, texts):
        counts = Counter()
        for text in texts:
            for word in text.split():
                counts.update(self._pieces(word))
        return counts

    def train(self, corpora, vocab_size: int, verbose: bool = False):
        """Learn joint merges.

        corpora: {"fr": [lines], "en": [lines]} (or a plain list of lines, which
        trains a joint vocab without per-language counts).
        """
        if not isinstance(corpora, dict):
            corpora = {None: list(corpora)}

        lang_words = {lang: self._word_counts(lines) for lang, lines in corpora.items()}
        joint = Counter()
        for counts in lang_words.values():
            joint.update(counts)

        # Each unique word as a mutable list of symbols, plus its frequency.
        words = [list(w) + [self.EOW] for w in joint]
        freqs = list(joint.values())

        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.base_tokens = {sym for w in words for sym in w}
        for sym in sorted(self.base_tokens):
            self.vocab.setdefault(sym, len(self.vocab))

        # Pair counts + which words contain each pair, updated incrementally
        # instead of recounting the whole corpus after every merge.
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        # Max-heap with lazy deletion: stale entries are skipped on pop.
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapify(heap)

        self.merges, self.split_of = [], {}
        while len(self.vocab) < vocab_size and heap:
            neg, pair = heappop(heap)
            count = pair_counts.get(pair, 0)
            if -neg != count:
                continue          # stale entry; the current count was pushed separately
            if count < 2:
                break

            a, b = pair
            merged = a + b
            self.merges.append(pair)
            self.split_of.setdefault(merged, pair)
            # Two different pairs can spell the same string ("ab"+"c", "a"+"bc");
            # keep its first id so ids stay contiguous.
            self.vocab.setdefault(merged, len(self.vocab))

            changed = set()
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                new_w, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        new_w.append(merged)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed.add(p)

            del where[pair]
            pair_counts.pop(pair, None)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)

            if verbose and len(self.merges) % 500 == 0:
                print(f"  merge {len(self.merges)}: {pair} -> {merged!r} (count={count})")

        self._finalize()

        # Per-language token counts under the final segmentation.
        self.lang_counts = {}
        for lang, counts in lang_words.items():
            if lang is None:
                continue
            tc = Counter()
            for word, f in counts.items():
                for tok in self._segment(word):
                    tc[tok] += f
            self.lang_counts[lang] = dict(tc)

        print(f"Training done. Vocab size: {len(self.vocab)}, merges: {len(self.merges)}")

    def _finalize(self):
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self._cache = {}

    # ------------------------------------------------------------- encoding
    def _segment(self, word):
        """Apply merges by rank (lowest first), like the original BPE."""
        syms = list(word) + [self.EOW]
        ranks = self.merge_ranks
        while len(syms) > 1:
            best_rank = None
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
            if best_rank is None:
                break
            a, b = self.merges[best_rank]
            # Merge every occurrence of this pair in one pass.
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        return syms

    def _split_rare(self, tok, allowed):
        """Undo merges until every piece is frequent enough in this language."""
        if tok in allowed or tok in self.base_tokens or tok not in self.split_of:
            return [tok]
        a, b = self.split_of[tok]
        return self._split_rare(a, allowed) + self._split_rare(b, allowed)

    def tokenize(self, text: str, lang=None, threshold: int = 0):
        allowed = None
        if lang is not None and threshold > 0 and lang in self.lang_counts:
            allowed = {t for t, c in self.lang_counts[lang].items() if c >= threshold}

        tokens = []
        for word in text.split():
            key = (word, lang, threshold)
            toks = self._cache.get(key)
            if toks is None:
                toks = [t for piece in self._pieces(word) for t in self._segment(piece)]
                if allowed is not None:
                    toks = [p for tok in toks for p in self._split_rare(tok, allowed)]
                if len(self._cache) < 500_000:
                    self._cache[key] = toks
            tokens.extend(toks)
        return tokens

    def encode(self, text: str, lang=None, threshold: int = 0, add_special: bool = False):
        """Text -> ids. No <s>/</s> unless add_special=True (matches SentencePiece)."""
        unk = self.unk_id
        ids = [self.vocab.get(t, unk) for t in self.tokenize(text, lang, threshold)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        skip = {self.pad_id, self.bos_id, self.eos_id}
        text = "".join(self.id_to_token.get(i, "<unk>") for i in ids if i not in skip)
        if self.pretokenize:
            return self._join(text)
        return text.replace(self.EOW, " ").strip()

    # ---------------------------------------------------------- persistence
    def save(self, path: str):
        data = {
            "version": 2,
            "pretokenize": self.pretokenize,
            "special_tokens": self.special_tokens,
            "merges": [list(p) for p in self.merges],
            "vocab": self.vocab,
            "lang_counts": self.lang_counts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Saved to {path}")

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        # Files trained before pre-tokenization existed must keep the old behaviour.
        self.pretokenize = data.get("pretokenize", False)
        if data.get("version", 1) >= 2:
            self.merges = [tuple(p) for p in data["merges"]]
        else:  # v1 files stored [[a, b], merged]; JSON lists keep merge order
            self.merges = [tuple(p) for p, _ in data["merges"]]
        self.lang_counts = data.get("lang_counts", {})
        self.split_of = {}
        for a, b in self.merges:
            self.split_of.setdefault(a + b, (a, b))
        self.base_tokens = {t for t in self.vocab
                            if t not in self.split_of and t not in self.special_tokens}
        self._finalize()
        print(f"Loaded vocab size: {len(self.vocab)}")
        return self
