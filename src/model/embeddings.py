# token + sinusoidal positional

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        # Scale by sqrt(d_model) as in the original paper
        return self.embedding(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding matrix once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, x):
        return self.position(self.token(x))