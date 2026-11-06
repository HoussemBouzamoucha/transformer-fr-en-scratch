# causal + padding masksimport torch
import torch

def make_padding_mask(seq, pad_id: int = 0):
    """
    seq: (batch, seq_len)
    Returns: (batch, 1, 1, seq_len) — True where token is NOT pad
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len: int, device=None):
    """
    Returns: (1, 1, seq_len, seq_len) — upper triangle is 0 (blocked)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_mask(tgt, pad_id: int = 0):
    """
    Combines causal mask and padding mask for decoder self-attention.
    tgt: (batch, seq_len)
    Returns: (batch, 1, seq_len, seq_len)
    """
    seq_len = tgt.size(1)
    causal = make_causal_mask(seq_len, device=tgt.device)
    padding = make_padding_mask(tgt, pad_id)
    return causal & padding