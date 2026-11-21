"""Build and call the multithreaded C++ encoder (fast_bpe.cpp) through ctypes.

Used by BPETokenizer.encode_batch(); the pure-Python encode() stays the
reference, and tests/test_fast_bpe.py checks both give the same ids.

The library is compiled on first use with g++ or clang++ (or $CXX) into
src/tokenizer/_build/, named after a hash of the source, so editing the .cpp
rebuilds it. No compiler -> get_encoder() returns None and encode_batch() falls
back to Python. Set BPE_FAST=0 to force the fallback.
"""
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np

from src.tokenizer.normalize import _CONTROL, _MAP

_SRC = Path(__file__).with_name("fast_bpe.cpp")
_BUILD = Path(__file__).with_name("_build")
_LIB = None
_LIB_ERROR = None
_LOCK = threading.Lock()
N_CP = 0x110000


def _flags():
    if sys.platform == "win32":  # MinGW: static runtime, so the .dll needs nothing else
        return ["-O3", "-std=c++17", "-shared", "-static"]
    return ["-O3", "-std=c++17", "-shared", "-fPIC", "-pthread"]


def _compiler():
    for c in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def _build():
    cxx = _compiler()
    if cxx is None:
        raise RuntimeError("no C++ compiler found (install g++, or set CXX)")
    flags = _flags()
    tag = hashlib.sha1(_SRC.read_bytes() + " ".join(flags).encode()).hexdigest()[:12]
    out = _BUILD / f"fast_bpe_{tag}{'.dll' if sys.platform == 'win32' else '.so'}"
    if not out.exists():
        _BUILD.mkdir(exist_ok=True)
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp{out.suffix}")
        r = subprocess.run([cxx, *flags, str(_SRC), "-o", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compiling {_SRC.name} failed:\n{r.stderr[-3000:]}")
        os.replace(tmp, out)  # atomic: parallel processes building at once is fine
    return out


def _load():
    global _LIB, _LIB_ERROR
    with _LOCK:
        if _LIB is None and _LIB_ERROR is None:
            try:
                lib = ctypes.CDLL(str(_build()))
                P, I32, I64, U32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_uint32
                lib.bpe_new.restype = P
                lib.bpe_new.argtypes = [P, P, I32, P, P, P, P, I32, P, I32, I32, U32, I32,
                                        P, P, P, I32]
                lib.bpe_free.argtypes = [P]
                lib.bpe_encode.restype = P
                lib.bpe_encode.argtypes = [P, ctypes.c_char_p, I64, I32, P, I64, I32, I32]
                lib.res_n_sent.restype = I64
                lib.res_n_sent.argtypes = [P]
                lib.res_n_ids.restype = I64
                lib.res_n_ids.argtypes = [P]
                lib.res_copy.argtypes = [P, P, P]
                lib.res_free.argtypes = [P]
                # training (FastTrainer)
                for name, res, args in (
                        ("tr_new", P, [P]), ("tr_free", None, [P]), ("tr_intern", I32, [P, P, I64]),
                        ("tr_count_text", None, [P, I32, ctypes.c_char_p, I64, I32, I32, I32]),
                        ("tr_char_counts", I64, [P]), ("tr_chars", None, [P, P, P]),
                        ("tr_merge", I64, [P, P, I32, I32, I64]), ("tr_merges", None, [P, P, P, P, P]),
                        ("tr_fit", I64, [P, P, I32, P, I32, I64, I64, I64, I32]),
                        ("tr_count_vocab", None, [P, P, I32, P, I32, I32]),
                        ("tr_n_kept", I64, [P]), ("tr_kept", None, [P, P]),
                        ("tr_n_counts", I64, [P, I32]), ("tr_counts", None, [P, I32, P, P]),
                        ("tr_n_syms", I64, [P]), ("tr_syms_len", I64, [P]), ("tr_syms", None, [P, P, P])):
                    fn = getattr(lib, name)
                    fn.restype, fn.argtypes = res, args
                _LIB = lib
            except Exception as e:  # no compiler, failed build...
                _LIB_ERROR = e
                warnings.warn(f"C++ BPE encoder unavailable, using Python ({e})")
    return _LIB


_CLASSES = None


def char_classes():
    """One byte per code point: 1 = re \\w, 2 = re \\s, 4 = str.isspace(),
    8 = bpe._wordchar, 16 = removed by normalize() (_CONTROL)."""
    global _CLASSES
    if _CLASSES is None:
        every = "".join(map(chr, range(N_CP)))
        cls = np.zeros(N_CP, dtype=np.uint8)
        for bit, pattern in ((1, re.compile(r"\w")), (2, re.compile(r"\s")), (16, _CONTROL)):
            idx = [m.start() for m in pattern.finditer(every)]
            cls[idx] |= bit
        cls[[c for c in range(N_CP) if every[c].isspace()]] |= 4
        cls[[c for c in range(N_CP) if every[c].isalnum() or c == 95]] |= 8
        _CLASSES = cls
    return _CLASSES


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _map_tables():
    """normalize._MAP (a str.maketrans table) as code point -> replacement code points."""
    cmap = {k: "" if v is None else v if isinstance(v, str) else chr(v) for k, v in _MAP.items()}
    map_from = np.array(list(cmap), dtype=np.uint32)
    map_cps = np.array([ord(c) for v in cmap.values() for c in v] or [0], dtype=np.uint32)
    map_offs = np.zeros(len(cmap) + 1, dtype=np.int64)
    np.cumsum([len(v) for v in cmap.values()], out=map_offs[1:])
    return map_from, map_offs, map_cps, len(cmap)


def _pack(strings):
    """Strings -> (code points, offsets) arrays for C++."""
    cps = np.frombuffer("".join(strings).encode("utf-32-le", "surrogatepass") or bytes(4),
                        dtype=np.uint32)
    offs = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in strings], out=offs[1:])
    return cps, offs


def _unpack(cps, offs):
    text = cps.tobytes().decode("utf-32-le", "surrogatepass")
    o = offs.tolist()
    return [text[o[i]:o[i + 1]] for i in range(len(o) - 1)]


class FastEncoder:
    """C++ copy of one BPETokenizer's encoding tables. Rebuild it after the
    tokenizer changes (BPETokenizer does that in _finalize)."""

    LANG = {"fr": 1, "en": 2}

    def __init__(self, bpe):
        self._lib = _load()
        if self._lib is None:
            raise RuntimeError(f"C++ BPE encoder unavailable: {_LIB_ERROR}")
        ids = {}

        def intern(s):
            if s not in ids:
                ids[s] = len(ids)
            return ids[s]

        for t in bpe.vocab:
            intern(t)
        merges = np.array([(intern(a), intern(b), intern(a + b)) for a, b in bpe.merges],
                          dtype=np.int32).reshape(-1)
        for tok, (a, b) in bpe.split_of.items():
            intern(tok), intern(a), intern(b)
        eow = intern(bpe.EOW)

        n = len(ids)
        vocab_id = np.full(n, -1, dtype=np.int32)
        split_a = np.full(n, -1, dtype=np.int32)
        split_b = np.full(n, -1, dtype=np.int32)
        for s, k in ids.items():
            if s in bpe.vocab:
                vocab_id[k] = bpe.vocab[s]
        for tok, (a, b) in bpe.split_of.items():
            split_a[ids[tok]], split_b[ids[tok]] = ids[a], ids[b]
        cps = np.array([ord(c) for s in ids for c in s], dtype=np.uint32)
        offs = np.zeros(n + 1, dtype=np.int64)
        np.cumsum([len(s) for s in ids], out=offs[1:])
        cls = char_classes()
        map_from, map_offs, map_cps, n_map = _map_tables()

        self._ids = ids
        self._masks = {}  # id(allowed frozenset) -> (frozenset, mask, cache_key)
        self._h = self._lib.bpe_new(_ptr(cps), _ptr(offs), n, _ptr(vocab_id), _ptr(split_a),
                                    _ptr(split_b), _ptr(merges), len(bpe.merges), _ptr(cls),
                                    eow, bpe.unk_id, ord(bpe.GLUE), int(bpe.pretokenize),
                                    _ptr(map_from), _ptr(map_offs), _ptr(map_cps), n_map)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.bpe_free(h)

    def _mask(self, allowed):
        if allowed is None:
            return None, -1
        hit = self._masks.get(id(allowed))
        if hit is None or hit[0] is not allowed:
            mask = np.zeros(len(self._ids), dtype=np.uint8)
            mask[[self._ids[t] for t in allowed if t in self._ids]] = 1
            self._next_key = getattr(self, "_next_key", 0) + 1  # new key = fresh C++ word cache
            hit = self._masks[id(allowed)] = (allowed, mask, self._next_key)
        return hit[1], hit[2]

    def encode(self, lines: str, n: int, lang, allowed, normalize: bool, num_threads=None):
        """`lines`: n sentences joined by "\\n", already NFC when `normalize` (the rest
        of normalize() is applied in C++). Returns n lists of ids."""
        mask, mask_key = self._mask(allowed)
        lang_code = self.LANG.get(lang, 0)
        buf = lines.encode("utf-8", "surrogatepass")
        threads = num_threads or os.cpu_count() or 1
        res = self._lib.bpe_encode(self._h, buf, len(buf), lang_code,
                                   None if mask is None else _ptr(mask),
                                   (lang_code << 32) | (mask_key & 0xFFFFFFFF), int(normalize),
                                   threads)
        try:
            if self._lib.res_n_sent(res) != n:
                raise RuntimeError(f"C++ encoder saw {self._lib.res_n_sent(res)} sentences, expected {n}")
            out_ids = np.empty(self._lib.res_n_ids(res), dtype=np.int32)
            out_offs = np.empty(n + 1, dtype=np.int64)
            self._lib.res_copy(res, _ptr(out_ids), _ptr(out_offs))
        finally:
            self._lib.res_free(res)
        flat, offs = out_ids.tolist(), out_offs.tolist()
        return [flat[offs[i]:offs[i + 1]] for i in range(n)]


def get_encoder(bpe):
    """FastEncoder for `bpe`, or None when C++ is unavailable or BPE_FAST=0."""
    if os.environ.get("BPE_FAST", "1") == "0" or _load() is None:
        return None
    return FastEncoder(bpe)


class FastTrainer:
    """C++ BPETokenizer.train() steps (fast_bpe.cpp, "Training"). BPETokenizer._train_fast
    drives it; tests/test_fast_bpe.py checks the result against _train_python."""

    LANG = FastEncoder.LANG

    def __init__(self, bpe):
        self._lib = _load()
        if self._lib is None:
            raise RuntimeError(f"C++ BPE trainer unavailable: {_LIB_ERROR}")
        lib = self._lib
        # A model with no vocab: only its character classes, _MAP, glue mark and
        # pretokenize setting are used, by the piece counting.
        empty_i32 = np.zeros(1, dtype=np.int32)
        empty_u32 = np.zeros(1, dtype=np.uint32)
        zero_off = np.zeros(1, dtype=np.int64)
        map_from, map_offs, map_cps, n_map = _map_tables()
        self._pre = lib.bpe_new(_ptr(empty_u32), _ptr(zero_off), 0, _ptr(empty_i32), _ptr(empty_i32),
                                _ptr(empty_i32), _ptr(empty_i32), 0, _ptr(char_classes()), -1, 0,
                                ord(bpe.GLUE), int(bpe.pretokenize),
                                _ptr(map_from), _ptr(map_offs), _ptr(map_cps), n_map)
        self._h = lib.tr_new(self._pre)

    def __del__(self):
        lib = getattr(self, "_lib", None)
        if getattr(self, "_h", None):
            lib.tr_free(self._h)
            self._h = None
        if getattr(self, "_pre", None):
            lib.bpe_free(self._pre)
            self._pre = None

    @staticmethod
    def _i32(xs):
        return np.array(list(xs) or [0], dtype=np.int32), len(xs)

    def count_text(self, slot, lines: str, lang, normalize: bool, threads: int):
        buf = lines.encode("utf-8", "surrogatepass")
        self._lib.tr_count_text(self._h, slot, buf, len(buf), self.LANG.get(lang, 0),
                                int(normalize), threads)

    def char_counts(self):
        n = self._lib.tr_char_counts(self._h)
        cps, cnt = np.zeros(max(n, 1), dtype=np.uint32), np.zeros(max(n, 1), dtype=np.int64)
        self._lib.tr_chars(self._h, _ptr(cps), _ptr(cnt))
        return {chr(c): int(k) for c, k in zip(cps[:n].tolist(), cnt[:n].tolist())}

    def intern(self, s: str) -> int:
        cps, _ = _pack([s])
        return self._lib.tr_intern(self._h, _ptr(cps), len(s))

    def merge(self, rare, eow_id: int, n_new_max: int):
        """Runs the merge loop; returns (a ids, b ids, merged ids, counts) per merge."""
        rare_cps = np.array([ord(c) for c in rare] or [0], dtype=np.uint32)
        n = self._lib.tr_merge(self._h, _ptr(rare_cps), len(rare), eow_id, n_new_max)
        out = [np.zeros(max(n, 1), dtype=t) for t in (np.int32, np.int32, np.int32, np.int64)]
        self._lib.tr_merges(self._h, *map(_ptr, out))
        return [o[:n].tolist() for o in out]

    def syms(self):
        n, total = self._lib.tr_n_syms(self._h), self._lib.tr_syms_len(self._h)
        cps, offs = np.zeros(max(total, 1), dtype=np.uint32), np.zeros(n + 1, dtype=np.int64)
        self._lib.tr_syms(self._h, _ptr(cps), _ptr(offs))
        return _unpack(cps[:total], offs)

    def fit(self, slots, base_ids, vocab_size, n_fixed, threshold, threads) -> int:
        sl, n_sl = self._i32(slots)
        base, n_base = self._i32(base_ids)
        return self._lib.tr_fit(self._h, _ptr(sl), n_sl, _ptr(base), n_base, vocab_size, n_fixed,
                                threshold, threads)

    def count_vocab(self, slots, vocab_ids, threads):
        sl, n_sl = self._i32(slots)
        voc, n_voc = self._i32(vocab_ids)
        self._lib.tr_count_vocab(self._h, _ptr(sl), n_sl, _ptr(voc), n_voc, threads)

    def kept(self):
        n = self._lib.tr_n_kept(self._h)
        out = np.zeros(max(n, 1), dtype=np.int32)
        self._lib.tr_kept(self._h, _ptr(out))
        return out[:n].tolist()

    def counts(self, i):
        """Token counts of fit slot i (the i-th of `slots`), in first-seen order."""
        n = self._lib.tr_n_counts(self._h, i)
        ids, vals = np.zeros(max(n, 1), dtype=np.int32), np.zeros(max(n, 1), dtype=np.int64)
        self._lib.tr_counts(self._h, i, _ptr(ids), _ptr(vals))
        return ids[:n].tolist(), vals[:n].tolist()


def get_trainer(bpe):
    """FastTrainer for `bpe`, or None when C++ is unavailable or BPE_FAST=0."""
    if os.environ.get("BPE_FAST", "1") == "0" or _load() is None:
        return None
    return FastTrainer(bpe)
