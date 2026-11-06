 # vs F.scaled_dot_product_attentionimport sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from src.model.attention import scaled_dot_product_attention, MultiHeadAttention

D_MODEL = 256
N_HEADS = 4
D_K = D_MODEL // N_HEADS
BATCH = 2
SEQ_LEN = 10


def test_sdpa_matches_pytorch():
    """Our scaled dot-product attention must match F.scaled_dot_product_attention within 1e-5."""
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)

    our_out, _ = scaled_dot_product_attention(q, k, v)
    pt_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    assert torch.allclose(our_out, pt_out, atol=1e-5), \
        f"Max diff: {(our_out - pt_out).abs().max().item()}"
    print("  scaled_dot_product_attention matches PyTorch: OK")


def test_sdpa_shape():
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_K)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (BATCH, N_HEADS, SEQ_LEN, D_K)
    assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)
    print(f"  sdpa output shape: {out.shape} OK")
    print(f"  sdpa weights shape: {weights.shape} OK")


def test_mha_matches_pytorch():
    """Copy weights into nn.MultiheadAttention and compare outputs."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)

    import torch.nn as nn
    ref = nn.MultiheadAttention(D_MODEL, N_HEADS, bias=False, batch_first=True)

    # Copy weights
    with torch.no_grad():
        ref.in_proj_weight[:D_MODEL].copy_(mha.W_q.weight)
        ref.in_proj_weight[D_MODEL:2*D_MODEL].copy_(mha.W_k.weight)
        ref.in_proj_weight[2*D_MODEL:].copy_(mha.W_v.weight)
        ref.out_proj.weight.copy_(mha.W_o.weight)

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    our_out, _ = mha(x, x, x)
    ref_out, _ = ref(x, x, x, need_weights=False)

    assert torch.allclose(our_out, ref_out, atol=1e-5), \
        f"Max diff: {(our_out - ref_out).abs().max().item()}"
    print("  MultiHeadAttention matches nn.MultiheadAttention: OK")


def test_mha_shape():
    mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out, weights = mha(x, x, x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    print(f"  mha output shape: {out.shape} OK")


if __name__ == "__main__":
    print("Testing attention...")
    test_sdpa_shape()
    test_sdpa_matches_pytorch()
    test_mha_shape()
    test_mha_matches_pytorch()
    print("\nAll attention tests passed.")