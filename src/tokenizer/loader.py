"""Build the tokenizer named in the config, behind one small interface.

Both backends expose: encode(text, lang) -> ids (no <s>/</s>), decode(ids),
pad_id, bos_id, eos_id, vocab_size.
"""
from src.tokenizer.bpe import BPETokenizer


class SentencePieceTokenizer:
    def __init__(self, path: str):
        import sentencepiece as spm  # only needed for this backend
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.piece_to_id("<s>")
        self.eos_id = self.sp.piece_to_id("</s>")
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, lang=None):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)


class JointBPETokenizer:
    """Wraps BPETokenizer with the per-language vocabulary threshold from the config."""

    def __init__(self, path: str, vocab_threshold: int = 0):
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


def load_tokenizer(cfg: dict):
    tok_cfg = cfg["tokenizer"]
    kind = tok_cfg.get("type", "sentencepiece")
    if kind == "bpe":
        return JointBPETokenizer(tok_cfg["path"], tok_cfg.get("vocab_threshold", 0))
    if kind == "sentencepiece":
        path = tok_cfg.get("path") or cfg["data"].get("sp_model", "data/vocab/spm_8k_wmt.model")
        return SentencePieceTokenizer(path)
    raise ValueError(f"Unknown tokenizer type: {kind!r} (expected 'bpe' or 'sentencepiece')")
