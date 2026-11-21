# encoder layer, decoder layer (Post-LN as in the paper, or Pre-LN)
import torch.nn as nn
from src.model.attention import MultiHeadAttention

NORM_POSITIONS = ("post", "pre")


def check_norm_position(norm_position: str):
    if norm_position not in NORM_POSITIONS:
        raise ValueError(f"norm_position must be one of {NORM_POSITIONS}, got {norm_position!r}")


def residual(x, sublayer, norm, dropout, pre: bool):
    """Residual + LayerNorm around one sublayer, dropout on the sublayer output.

    post (paper, sec. 3.1 / 5.4): LayerNorm(x + Dropout(Sublayer(x)))
    pre  (Xiong et al. 2020):     x + Dropout(Sublayer(LayerNorm(x)))
    """
    if pre:
        return x + dropout(sublayer(norm(x)))
    return norm(x + dropout(sublayer(x)))


class FeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2  (Vaswani et al. 2017, eq. 2).

    The dropout between the ReLU and W2 is not in the paper (its code sets this
    relu_dropout to 0); model.ffn_dropout: 0.0 turns it off.
    """
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
    # norm_position defaults to "pre" because that is what checkpoints saved
    # before the option existed were trained with.
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_position: str = "pre", ffn_dropout: float = None):
        super().__init__()
        check_norm_position(norm_position)
        self.pre = norm_position == "pre"
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout if ffn_dropout is None else ffn_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = residual(x, lambda h: self.self_attn(h, h, h, mask=src_mask)[0],
                     self.norm1, self.dropout, self.pre)
        return residual(x, self.ff, self.norm2, self.dropout, self.pre)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_position: str = "pre", ffn_dropout: float = None):
        super().__init__()
        check_norm_position(norm_position)
        self.pre = norm_position == "pre"
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout if ffn_dropout is None else ffn_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        # Masked self-attention, then attention over the encoder output, then FFN.
        x = residual(x, lambda h: self.self_attn(h, h, h, mask=tgt_mask)[0],
                     self.norm1, self.dropout, self.pre)
        x = residual(x, lambda h: self.cross_attn(h, memory, memory, mask=src_mask)[0],
                     self.norm2, self.dropout, self.pre)
        return residual(x, self.ff, self.norm3, self.dropout, self.pre)
