"""Load a trained checkpoint once and translate your own French sentences.

    from src.inference.translator import Translator
    tr = Translator("configs/thin_slice.yaml")            # checkpoints/thin_slice/best.pt
    tr("Un chat noir dort sur le canapé.")                # -> "A black cat is sleeping on the couch."
    tr(["Bonjour.", "Deux enfants jouent au football."])  # list in, list out
    tr.explain("Un chat noir dort.")                      # also shows the BPE tokens
"""
from pathlib import Path

import torch

from src.eval.metrics import translate
from src.model.transformer import Transformer
from src.tokenizer.loader import check_fingerprint, load_tokenizer
from src.utils import load_config


def load_model(config_path, checkpoint=None, device=None, force=False):
    """Build the model from the checkpoint's config and verify the tokenizer matches."""
    cfg = load_config(config_path)
    ckpt_path = Path(checkpoint or f"checkpoints/{Path(config_path).stem}/best.pt")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = load_tokenizer(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not force:
        check_fingerprint(ckpt.get("tokenizer"), tokenizer)

    m = ckpt.get("config", cfg)["model"]
    model = Transformer(
        vocab_size=tokenizer.vocab_size, d_model=m["d_model"], n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"], n_decoder_layers=m["n_decoder_layers"],
        d_ff=m["d_ff"], max_len=m["max_seq_len"], dropout=0.0, pad_id=tokenizer.pad_id,
        tie_embeddings=m.get("shared_vocab", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, cfg, ckpt, ckpt_path, device


class Translator:
    def __init__(self, config_path="configs/thin_slice.yaml", checkpoint=None, device=None,
                 restrict_output=True, force=False):
        (self.model, self.tokenizer, self.cfg, ckpt,
         self.ckpt_path, self.device) = load_model(config_path, checkpoint, device, force)
        self.max_seq_len = ckpt.get("config", self.cfg)["model"]["max_seq_len"]
        self.restrict_output = restrict_output
        print(f"Loaded {self.ckpt_path} (step {ckpt.get('step')}, val BLEU {ckpt.get('val_bleu')}) "
              f"on {self.device}")

    def __call__(self, text, batch_size=64):
        lines = [text] if isinstance(text, str) else list(text)
        out = translate(self.model, self.tokenizer, lines, self.device, batch_size=batch_size,
                        max_len=self.max_seq_len - 2, max_src_len=self.max_seq_len,
                        restrict_output=self.restrict_output)
        return out[0] if isinstance(text, str) else out

    def explain(self, text):
        """Print the French tokens, the English translation and its tokens."""
        out = self(text)
        tok = self.tokenizer
        if hasattr(tok, "bpe"):
            src_toks = tok.bpe.tokenize(text, lang="fr", threshold=tok.threshold)
            out_toks = tok.bpe.tokenize(out, lang="en", threshold=tok.threshold)
            unk = [t for t in src_toks if t not in tok.bpe.vocab]
            print(f"FR  {text}\n    {src_toks}" + (f"\n    unknown -> <unk>: {unk}" if unk else ""))
            print(f"EN  {out}\n    {out_toks}")
        else:
            print(f"FR  {text}\nEN  {out}")
        return out
