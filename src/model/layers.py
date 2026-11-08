#encoder layer, decoder layer (Pre-LN)import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-LN: normalize before sublayer
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        # Masked self-attention (Pre-LN)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=tgt_mask)
        x = x + self.dropout(attn_out)

        # Cross-attention (Pre-LN)
        attn_out, _ = self.cross_attn(self.norm2(x), memory, memory, mask=src_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward (Pre-LN)
        ff_out = self.ff(self.norm3(x))
        x = x + self.dropout(ff_out)
        return x