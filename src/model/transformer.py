# full encoder-decoder modelimport torch
import torch.nn as nn
from src.model.embeddings import Embeddings
from src.model.layers import EncoderLayer, DecoderLayer
from src.model.masks import make_padding_mask, make_causal_mask, make_decoder_mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.embeddings(src)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

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
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(vocab_size, d_model, n_heads, n_encoder_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_heads, n_decoder_layers, d_ff, max_len, dropout)

        # Tied output projection — shares weights with decoder token embedding
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