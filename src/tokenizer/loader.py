"""Build the tokenizer named in the config, behind one small interface.

Both backends expose: encode(text, lang) -> ids (no <s>/</s>), decode(ids),
pad_id, bos_id, eos_id, vocab_size, target_token_ids(), fingerprint().
"""
import hashlib
from pathlib import Path

from src.tokenizer.bpe import BPETokenizer


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SentencePieceTokenizer:
    kind = "sentencepiece"

    def __init__(self, path: str):
        import sentencepiece as spm  # only needed for this backend
        self.path = path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.piece_to_id("<s>")
        self.eos_id = self.sp.piece_to_id("</s>")
        self.vocab_size = self.sp.get_piece_size()
        self.threshold = 0

    def encode(self, text: str, lang=None):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)

    def target_token_ids(self, lang="en"):
        return None  # no per-language counts

    def fingerprint(self):
        return {"type": self.kind, "path": str(self.path), "sha256": _sha256(self.path),
                "vocab_size": self.vocab_size}


class HFTokenizer:
    """A HuggingFace `tokenizers` file (tokenizer.json), e.g. the retrained GPT-2 byte-level BPE."""
    kind = "hf"

    def __init__(self, path: str):
        from tokenizers import Tokenizer  # only needed for this backend
        self.path = path
        self.tok = Tokenizer.from_file(path)
        self.pad_id = self.tok.token_to_id("<pad>")
        self.bos_id = self.tok.token_to_id("<s>")
        self.eos_id = self.tok.token_to_id("</s>")
        self.vocab_size = self.tok.get_vocab_size()
        self.threshold = 0

    def encode(self, text: str, lang=None):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def target_token_ids(self, lang="en"):
        return None

    def fingerprint(self):
        return {"type": self.kind, "path": str(self.path), "sha256": _sha256(self.path),
                "vocab_size": self.vocab_size}


class JointBPETokenizer:
    """Wraps BPETokenizer with the per-language vocabulary threshold from the config."""
    kind = "bpe"

    def __init__(self, path: str, vocab_threshold: int = 0):
        self.path = path
        self.bpe = BPETokenizer().load(path)
        self.threshold = vocab_threshold
        self.pad_id = self.bpe.pad_id
        self.bos_id = self.bpe.bos_id
        self.eos_id = self.bpe.eos_id
        self.vocab_size = self.bpe.vocab_size

    def encode(self, text: str, lang=None):
        return self.bpe.encode(text, lang=lang, threshold=self.threshold)

    def decode(self, ids):
        return self.bpe.decode(ids)

    def target_token_ids(self, lang="en"):
        return self.bpe.target_token_ids(lang, self.threshold)

    def fingerprint(self):
        return {"type": self.kind, "path": str(self.path), "sha256": _sha256(self.path),
                "vocab_size": self.vocab_size, "vocab_threshold": self.threshold}


def load_tokenizer(cfg: dict):
    tok_cfg = cfg["tokenizer"]
    kind = tok_cfg.get("type", "sentencepiece")
    if kind == "bpe":
        path = tok_cfg["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"BPE vocab {path} not found. Build it with scripts/train_own_bpe.py")
        return JointBPETokenizer(path, tok_cfg.get("vocab_threshold", 0))
    if kind == "sentencepiece":
        path = tok_cfg.get("path") or cfg["data"].get("sp_model", "data/vocab/spm_8k_wmt.model")
        return SentencePieceTokenizer(path)
    if kind == "hf":
        return HFTokenizer(tok_cfg["path"])
    raise ValueError(f"Unknown tokenizer type: {kind!r} (expected 'bpe', 'sentencepiece' or 'hf')")


def check_fingerprint(saved: dict, tokenizer):
    """Refuse to pair a checkpoint with a different vocab: ids would mean other tokens."""
    if not saved:
        raise ValueError("Checkpoint has no tokenizer fingerprint (saved before this check "
                         "existed); its vocab can't be verified. Retrain, or pass --force.")
    current = tokenizer.fingerprint()
    bad = [k for k in ("type", "sha256", "vocab_size", "vocab_threshold")
           if saved.get(k) != current.get(k)]
    if bad:
        detail = ", ".join(f"{k}: checkpoint={saved.get(k)!r} now={current.get(k)!r}" for k in bad)
        raise ValueError(f"Tokenizer does not match the checkpoint ({detail}).")
