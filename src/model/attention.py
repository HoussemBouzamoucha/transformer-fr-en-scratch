# scaled dot-product, multi-head
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: (batch, heads, seq_len, d_k)
    mask:    (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        # x: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        # x: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        x, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        x = self.merge_heads(x)
        x = self.W_o(x)
        return x, attn_weights