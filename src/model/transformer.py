# full encoder-decoder modelimport torch
import torch.nn as nn
from src.model.embeddings import Embeddings
from src.model.layers import EncoderLayer, DecoderLayer
from src.model.masks import make_padding_mask, make_causal_mask, make_decoder_mask


def _final_norm(d_model, norm_position):
    # Pre-LN leaves the residual stream unnormalized, so it needs one LayerNorm
    # at the end of the stack. Post-LN (the paper) already ends on a LayerNorm.
    return nn.LayerNorm(d_model) if norm_position == "pre" else nn.Identity()


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout,
                 norm_position="pre", ffn_dropout=None):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_position, ffn_dropout)
            for _ in range(n_layers)
        ])
        self.norm = _final_norm(d_model, norm_position)

    def forward(self, src, src_mask=None):
        x = self.embeddings(src)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout,
                 norm_position="pre", ffn_dropout=None):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_position, ffn_dropout)
            for _ in range(n_layers)
        ])
        self.norm = _final_norm(d_model, norm_position)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        x = self.embeddings(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 2,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_embeddings: bool = True,
        norm_position: str = "pre",
        ffn_dropout: float = None,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(vocab_size, d_model, n_heads, n_encoder_layers, d_ff, max_len, dropout,
                               norm_position, ffn_dropout)
        self.decoder = Decoder(vocab_size, d_model, n_heads, n_decoder_layers, d_ff, max_len, dropout,
                               norm_position, ffn_dropout)

        # Three-way tying (Vaswani et al. 2017, sec. 3.4): with a joint FR/EN
        # vocabulary, the encoder input embedding, decoder input embedding and
        # output projection are one matrix. Turn off for separate vocabularies.
        if tie_embeddings:
            self.decoder.embeddings.token = self.encoder.embeddings.token

        # The output projection always shares the decoder token embedding.
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.decoder.embeddings.token.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        src_mask = make_padding_mask(src, self.pad_id)
        return self.encoder(src, src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = make_decoder_mask(tgt, self.pad_id)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        decoder_out = self.decode(tgt, memory, src_mask)
        logits = self.output_projection(decoder_out)
        return logits


# Architecture switches. Each value is what a config that doesn't mention the key
# gets, i.e. what checkpoints saved before the switch existed were trained with,
# so every old checkpoint rebuilds exactly. New variants are opt-in per config.
# Add a new option here (e.g. pos_encoding: rope) next to the old one, never
# instead of it, so earlier experiments stay reproducible.
ARCH_DEFAULTS = {
    "norm_position": "pre",        # post (paper) | pre
    "pos_encoding": "sinusoidal",  # sinusoidal (paper)
    "shared_vocab": True,          # three-way embedding tying (paper sec. 3.4)
    "ffn_dropout": None,           # dropout inside the FFN; None = same as dropout, paper = 0.0
}
ARCH_CHOICES = {
    "norm_position": ("post", "pre"),
    "pos_encoding": ("sinusoidal",),
}


def build_model(model_cfg: dict, vocab_size: int, pad_id: int, dropout: float = None) -> Transformer:
    """Build the Transformer a config's `model:` block describes.

    Used by training and by checkpoint loading (with the config saved inside the
    checkpoint), so a checkpoint always rebuilds the architecture it was trained
    with. dropout overrides the config value (0.0 for inference).
    """
    m = {**ARCH_DEFAULTS, **model_cfg}
    for key, allowed in ARCH_CHOICES.items():
        if m[key] not in allowed:
            raise ValueError(f"model.{key} must be one of {allowed}, got {m[key]!r}")
    return Transformer(
        vocab_size       = vocab_size,
        d_model          = m["d_model"],
        n_heads          = m["n_heads"],
        n_encoder_layers = m["n_encoder_layers"],
        n_decoder_layers = m["n_decoder_layers"],
        d_ff             = m["d_ff"],
        max_len          = m["max_seq_len"],
        dropout          = m["dropout"] if dropout is None else dropout,
        pad_id           = pad_id,
        tie_embeddings   = m["shared_vocab"],
        norm_position    = m["norm_position"],
        # Inference passes dropout=0.0, which must switch this off too.
        ffn_dropout      = m["ffn_dropout"] if dropout is None else dropout,
    )


def describe_architecture(model_cfg: dict) -> str:
    m = {**ARCH_DEFAULTS, **model_cfg}
    return ", ".join(f"{k}={m[k]}" for k in ARCH_DEFAULTS)
