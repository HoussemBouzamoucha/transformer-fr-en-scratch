# overfit one batchimport sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.model.transformer import Transformer

VOCAB_SIZE = 8000
D_MODEL = 256
BATCH = 4
SRC_LEN = 20
TGT_LEN = 15


def test_model_output_shape():
    model = Transformer(VOCAB_SIZE, d_model=D_MODEL, n_heads=2,
                        n_encoder_layers=2, n_decoder_layers=2,
                        d_ff=512, dropout=0.0)
    src = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN))
    tgt = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN))
    logits = model(src, tgt)
    assert logits.shape == (BATCH, TGT_LEN, VOCAB_SIZE), f"Wrong shape: {logits.shape}"
    print(f"  model output shape: {logits.shape} OK")


def test_tied_embeddings():
    model = Transformer(VOCAB_SIZE, d_model=D_MODEL, n_heads=2,
                        n_encoder_layers=2, n_decoder_layers=2,
                        d_ff=512, dropout=0.0)
    assert model.output_projection.weight is \
           model.decoder.embeddings.token.embedding.weight, \
           "Output projection is not tied to decoder embeddings"
    print("  tied embeddings: OK")


def test_overfit_one_batch():
    """The most important test — model must drive loss to near zero on one batch."""
    torch.manual_seed(42)
    model = Transformer(VOCAB_SIZE, d_model=D_MODEL, n_heads=2,
                        n_encoder_layers=2, n_decoder_layers=2,
                        d_ff=512, dropout=0.0)

    src = torch.randint(1, VOCAB_SIZE, (4, 20))
    tgt = torch.randint(1, VOCAB_SIZE, (4, 15))

    # Teacher forcing: input is tgt[:-1], target is tgt[1:]
    tgt_in  = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    initial_loss = None

    for step in range(300):
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step % 50 == 0:
            print(f"    step {step:3d} loss: {loss.item():.4f}")

    final_loss = loss.item()
    print(f"  initial loss: {initial_loss:.4f} → final loss: {final_loss:.4f}")
    assert final_loss < 0.1, f"Model failed to overfit: final loss {final_loss:.4f}"
    print("  overfit one batch: OK")


if __name__ == "__main__":
    print("Testing full model...")
    test_model_output_shape()
    test_tied_embeddings()
    test_overfit_one_batch()
    print("\nAll model tests passed.")