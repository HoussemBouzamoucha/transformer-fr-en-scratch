#shape + gradient flowimport sys
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.layers import EncoderLayer, DecoderLayer

D_MODEL = 256
N_HEADS = 4
D_FF = 512
BATCH = 2
SEQ_LEN = 10


def test_encoder_layer_shape():
    layer = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = layer(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    print(f"  encoder layer shape: {out.shape} OK")


def test_decoder_layer_shape():
    layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    memory = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = layer(x, memory)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    print(f"  decoder layer shape: {out.shape} OK")


def test_encoder_gradients_flow():
    layer = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  encoder gradients flow to all parameters: OK")


def test_decoder_gradients_flow():
    layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    memory = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = layer(x, memory)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    assert memory.grad is not None, "No gradient at memory"
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  decoder gradients flow to all parameters: OK")


if __name__ == "__main__":
    print("Testing layers...")
    test_encoder_layer_shape()
    test_decoder_layer_shape()
    test_encoder_gradients_flow()
    test_decoder_gradients_flow()
    print("\nAll layer tests passed.")