# causal leakage testimport sys
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.masks import make_padding_mask, make_causal_mask, make_decoder_mask
from src.model.attention import MultiHeadAttention

D_MODEL = 256
N_HEADS = 4


def test_padding_mask_shape():
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = make_padding_mask(seq)
    assert mask.shape == (2, 1, 1, 5)
    print(f"  padding mask shape: {mask.shape} OK")


def test_padding_mask_values():
    seq = torch.tensor([[1, 2, 3, 0, 0]])
    mask = make_padding_mask(seq)
    expected = torch.tensor([[[[True, True, True, False, False]]]])
    assert torch.equal(mask, expected)
    print("  padding mask values: OK")


def test_causal_mask_shape():
    mask = make_causal_mask(5)
    assert mask.shape == (1, 1, 5, 5)
    print(f"  causal mask shape: {mask.shape} OK")


def test_causal_mask_values():
    mask = make_causal_mask(4)
    # Lower triangle should be 1, upper should be 0
    expected = torch.tensor([[[[1, 0, 0, 0],
                                [1, 1, 0, 0],
                                [1, 1, 1, 0],
                                [1, 1, 1, 1]]]], dtype=torch.float32)
    assert torch.equal(mask, expected)
    print("  causal mask values: OK")


def test_causal_leakage():
    """
    Changing a future token must NOT change earlier outputs.
    This is the critical leakage test from the plan.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
    seq_len = 6

    x1 = torch.randn(1, seq_len, D_MODEL)
    x2 = x1.clone()
    x2[0, 4, :] = torch.randn(D_MODEL)  # change token at position 4

    mask = make_causal_mask(seq_len)

    out1, _ = mha(x1, x1, x1, mask=mask)
    out2, _ = mha(x2, x2, x2, mask=mask)

    # Positions 0-3 must be identical (future change must not leak back)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "Causal leakage detected — future tokens affected earlier outputs"
    print("  causal leakage test: OK")


def test_pad_tokens_do_not_affect_output():
    """
    Pad tokens must not affect non-pad outputs.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)

    seq = torch.tensor([[1, 2, 3, 0, 0]])
    x1 = torch.randn(1, 5, D_MODEL)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(D_MODEL)  # change a pad position
    x2[0, 4, :] = torch.randn(D_MODEL)

    mask = make_padding_mask(seq)

    out1, _ = mha(x1, x1, x1, mask=mask)
    out2, _ = mha(x2, x2, x2, mask=mask)

    # Non-pad positions (0-2) must be identical
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5), \
        "Pad tokens affected non-pad outputs"
    print("  pad tokens do not affect non-pad outputs: OK")


def test_decoder_mask_shape():
    tgt = torch.tensor([[2, 5, 7, 0, 0]])
    mask = make_decoder_mask(tgt)
    assert mask.shape == (1, 1, 5, 5)
    print(f"  decoder mask shape: {mask.shape} OK")


if __name__ == "__main__":
    print("Testing masks...")
    test_padding_mask_shape()
    test_padding_mask_values()
    test_causal_mask_shape()
    test_causal_mask_values()
    test_causal_leakage()
    test_pad_tokens_do_not_affect_output()
    test_decoder_mask_shape()
    print("\nAll mask tests passed.")