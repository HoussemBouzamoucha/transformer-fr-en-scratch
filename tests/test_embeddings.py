# shape + scale (sqrt d_model)import sys
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import math
from src.model.embeddings import TokenEmbedding, SinusoidalPositionalEncoding, Embeddings

VOCAB_SIZE = 8000
D_MODEL = 256
BATCH = 4
SEQ_LEN = 20

def test_token_embedding_shape():
    emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    out = emb(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    print(f"  token embedding shape: {out.shape} OK")

def test_token_embedding_scale():
    emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    x = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
    out = emb(x)
    raw = emb.embedding(x)
    expected = raw * math.sqrt(D_MODEL)
    assert torch.allclose(out, expected), "Scaling by sqrt(d_model) failed"
    print(f"  token embedding scale (sqrt d_model): OK")

def test_positional_encoding_shape():
    pe = SinusoidalPositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
    out = pe(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    print(f"  positional encoding shape: {out.shape} OK")

def test_positional_encoding_values():
    pe = SinusoidalPositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.zeros(1, SEQ_LEN, D_MODEL)
    out = pe(x)
    # Position 0, dim 0 should be sin(0) = 0
    assert abs(out[0, 0, 0].item()) < 1e-5, "sin(0) should be 0"
    # Position 0, dim 1 should be cos(0) = 1
    assert abs(out[0, 0, 1].item() - 1.0) < 1e-5, "cos(0) should be 1"
    print(f"  positional encoding values: OK")

def test_full_embeddings_shape():
    emb = Embeddings(VOCAB_SIZE, D_MODEL, dropout=0.0)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    out = emb(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    print(f"  full embeddings shape: {out.shape} OK")

if __name__ == "__main__":
    print("Testing embeddings...")
    test_token_embedding_shape()
    test_token_embedding_scale()
    test_positional_encoding_shape()
    test_positional_encoding_values()
    test_full_embeddings_shape()
    print("\nAll embedding tests passed.")