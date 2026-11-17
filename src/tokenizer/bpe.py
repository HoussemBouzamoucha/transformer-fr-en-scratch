"""Joint BPE tokenizer (Sennrich et al., 2016) for French -> English translation.

One set of merges and one vocabulary are learned from both languages together.
Each language also keeps its own token counts, which are used to
  - undo merges seen (almost) only in the other language when encoding
    (the "vocabulary threshold" from subword-nmt), and
  - restrict which tokens the decoder may generate (target_token_ids).

Pipeline for one sentence:
  normalize (NFC, quotes, spaces)  ->  split on whitespace
  -> pre-tokenize each word (punctuation off, language-aware apostrophes, join marks)
  -> BPE merges by rank  ->  optional per-language threshold split
"""
import json
import re
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush

from src.tokenizer.normalize import normalize as normalize_text

# Letters/digits stay together; every other non-space character is its own piece.
_PIECE_RE = re.compile(r"\w+|[^\w\s]")

# Never mapped to <unk>, however rare: ASCII printable, Latin-1 (ñ ° £ § ...),
# the French letters outside Latin-1 (œ Œ Ÿ) and €.
ALWAYS_KEEP = (set(chr(c) for c in range(0x20, 0x7F)) | set(chr(c) for c in range(0xA1, 0x100))
               | set("œŒŸ€"))
ALWAYS_KEEP.discard("­")  # soft hyphen (normalize() removes it anyway)

PRETOKENIZE_VERSION = 2  # 0 = whitespace only; 1 = first punctuation rules (retired)


def _wordchar(c):
    return c.isalnum() or c == "_"


def _is_word(piece):
    return _wordchar(piece[0]) and _wordchar(piece[-1])


class BPETokenizer:
    EOW = "</w>"
    # Join mark: "⁐." = no space before this punctuation, "l'⁐" = no space after.
    # Each boundary inside a word gets exactly one mark, and word pieces never
    # carry one, so "fenêtre" is the same token in "fenêtre." and "fenêtre ouverte".
    GLUE = "⁐"

    def __init__(self, pretokenize: int = PRETOKENIZE_VERSION, normalize: bool = True):
        self.pretokenize = int(pretokenize)
        self.normalize = normalize
        self.min_char_count = 1
        self.merges = []          # ordered list of (a, b); index = merge rank
        self.merge_ranks = {}     # (a, b) -> rank
        self.split_of = {}        # merged token -> (a, b) that first produced it
        self.vocab = {}           # token -> id
        self.id_to_token = {}     # id -> token
        self.base_tokens = set()  # kept characters + </w>; never split further
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
    def _clean(self, text):
        return normalize_text(text) if self.normalize else text

    def _pieces(self, word, lang=None):
        """Split one whitespace-delimited word into BPE input pieces."""
        if self.pretokenize == 0:
            return [word]
        parts = _PIECE_RE.findall(word.replace(self.GLUE, ""))
        if len(parts) <= 1:
            return parts

        # Apostrophe between two words: French elision keeps it on the left
        # ("l'" "herbe", "qu'" "il"); English clitics keep it on the right
        # ("man" "'s", "don" "'t"). Unknown language: leave it standalone.
        if lang in ("fr", "en"):
            merged, i = [], 0
            while i < len(parts):
                p = parts[i]
                if (p == "'" and merged and _is_word(merged[-1])
                        and i + 1 < len(parts) and _is_word(parts[i + 1])):
                    if lang == "fr":
                        merged[-1] += "'"
                        i += 1
                    else:
                        merged.append("'" + parts[i + 1])
                        i += 2
                    continue
                merged.append(p)
                i += 1
            parts = merged

        # One join mark per boundary, on the right piece when it starts with
        # punctuation, otherwise on the left piece (which then ends with it).
        out = list(parts)
        for i in range(len(parts) - 1):
            if not _wordchar(parts[i + 1][0]):
                out[i + 1] = self.GLUE + out[i + 1]
            else:
                out[i] = out[i] + self.GLUE
        return out

    def _join(self, text):
        """Inverse of _pieces over a decoded string ("a</w>⁐.</w>" -> "a.")."""
        pieces = text.split(self.EOW)
        if pieces and pieces[-1] == "":
            pieces.pop()
        out = []
        for i, p in enumerate(pieces):
            if i > 0 and not (pieces[i - 1].endswith(self.GLUE) or p.startswith(self.GLUE)):
                out.append(" ")
            out.append(p)
        return "".join(out).replace(self.GLUE, "").strip()

    # ------------------------------------------------------------- training
    def _piece_counts(self, texts, lang):
        counts = Counter()
        for text in texts:
            for word in self._clean(text).split():
                counts.update(self._pieces(word, lang))
        return counts

    def train(self, corpora, vocab_size: int, min_char_count: int = 1, verbose: bool = False):
        """Learn joint merges.

        corpora: {"fr": [lines], "en": [lines]} (or a plain list of lines, which
        trains without per-language counts).
        min_char_count: characters seen fewer times (and not in ALWAYS_KEEP) get
        no vocab entry and encode as <unk>, so web-text junk doesn't use up slots.
        """
        if not isinstance(corpora, dict):
            corpora = {None: list(corpora)}
        self.min_char_count = min_char_count

        lang_pieces = {lang: self._piece_counts(lines, lang) for lang, lines in corpora.items()}
        joint = Counter()
        for counts in lang_pieces.values():
            joint.update(counts)

        char_counts = Counter()
        for piece, f in joint.items():
            for ch in piece:
                char_counts[ch] += f
        rare = {ch for ch, c in char_counts.items() if c < min_char_count and ch not in ALWAYS_KEEP}

        # Each unique piece as a mutable list of symbols, plus its frequency.
        words = [list(w) + [self.EOW] for w in joint]
        freqs = list(joint.values())

        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        # ALWAYS_KEEP characters get an entry even if this sample never had them,
        # so ordinary punctuation, digits and accents never become <unk>.
        self.base_tokens = ({sym for w in words for sym in w} - rare) | (ALWAYS_KEEP - {" "})
        for sym in sorted(self.base_tokens):
            self.vocab.setdefault(sym, len(self.vocab))

        def pairs_of(w):
            return [p for p in zip(w, w[1:]) if p[0] not in rare and p[1] not in rare]

        # Pair counts + which words contain each pair, updated incrementally
        # instead of recounting the whole corpus after every merge.
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in pairs_of(w):
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
                for p in pairs_of(w):
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
                for p in pairs_of(new_w):
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
        for lang, counts in lang_pieces.items():
            if lang is None:
                continue
            tc = Counter()
            for piece, f in counts.items():
                for tok in self._segment(piece):
                    if tok in self.vocab:
                        tc[tok] += f
            self.lang_counts[lang] = dict(tc)

        print(f"Training done. Vocab size: {len(self.vocab)}, merges: {len(self.merges)}, "
              f"rare chars -> <unk>: {len(rare)}")

    def _finalize(self):
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self._cache = {}

    # ------------------------------------------------------------- encoding
    def _segment(self, piece):
        """Apply merges by rank (lowest first), like the original BPE."""
        syms = list(piece) + [self.EOW]
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

    def _allowed(self, lang, threshold):
        if lang is None or threshold <= 0 or lang not in self.lang_counts:
            return None
        return {t for t, c in self.lang_counts[lang].items() if c >= threshold}

    def _split_rare(self, tok, allowed):
        """Undo merges until every piece is frequent enough in this language."""
        if tok in allowed or tok in self.base_tokens or tok not in self.split_of:
            return [tok]
        a, b = self.split_of[tok]
        return self._split_rare(a, allowed) + self._split_rare(b, allowed)

    def tokenize(self, text: str, lang=None, threshold: int = 0):
        allowed = self._allowed(lang, threshold)
        tokens = []
        for word in self._clean(text).split():
            key = (word, lang, threshold)
            toks = self._cache.get(key)
            if toks is None:
                toks = [t for piece in self._pieces(word, lang) for t in self._segment(piece)]
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

    def target_token_ids(self, lang: str = "en", threshold: int = 0):
        """Ids the decoder may emit for `lang`: tokens seen there (>= threshold
        times), every base character, </s> and <unk>. Everything a training
        target encoded with the same lang/threshold can contain is included."""
        if lang not in self.lang_counts:
            return None
        allowed = self._allowed(lang, max(threshold, 1))
        ids = {self.vocab[t] for t in allowed if t in self.vocab}
        ids |= {self.vocab[t] for t in self.base_tokens}
        ids |= {self.eos_id, self.unk_id}
        return sorted(ids)

    # ---------------------------------------------------------- persistence
    def save(self, path: str):
        data = {
            "version": 3,
            "pretokenize": self.pretokenize,
            "normalize": self.normalize,
            "min_char_count": self.min_char_count,
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
        pre = int(data.get("pretokenize", 0))  # True/False in older files
        if pre == 1:
            raise ValueError(f"{path} was built with the first punctuation rules, which were "
                             "replaced. Retrain it with scripts/train_own_bpe.py.")
        self.pretokenize = pre
        self.normalize = data.get("normalize", False)
        self.min_char_count = data.get("min_char_count", 1)
        self.vocab = data["vocab"]
        self.special_tokens = data.get("special_tokens", self.special_tokens)
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
