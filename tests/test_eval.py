import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.eval.metrics import evaluate_bleu, translate
from src.inference.greedy import greedy_decode
from src.model.transformer import Transformer
from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.loader import JointBPETokenizer, check_fingerprint

FR = ["le chat mange le poisson.", "le chien mange la viande.", "les chats dorment."] * 20
EN = ["the cat eats the fish.", "the dog eats the meat.", "the cats sleep."] * 20


@pytest.fixture
def tok(tmp_path):
    bpe = BPETokenizer()
    bpe.train({"fr": FR, "en": EN}, vocab_size=420)
    path = tmp_path / "bpe.json"
    bpe.save(str(path))
    return JointBPETokenizer(str(path), vocab_threshold=1)


def _model(vocab):
    torch.manual_seed(0)
    return Transformer(vocab, d_model=32, n_heads=2, n_encoder_layers=1,
                       n_decoder_layers=1, d_ff=64, max_len=64, dropout=0.0)


def test_greedy_respects_allowed_ids(tok):
    model = _model(tok.vocab_size)
    allowed = tok.target_token_ids("en")
    src = torch.tensor([[tok.bos_id] + tok.encode("le chat", lang="fr") + [tok.eos_id]])
    out = greedy_decode(model, src, tok.bos_id, tok.eos_id, max_len=20, allowed_ids=allowed)
    assert set(out.flatten().tolist()) <= set(allowed)


def test_translate_keeps_order_and_count(tok):
    model = _model(tok.vocab_size)
    lines = ["le chat", "les chats dorment le soir avec le chien", "le"]
    hyps = translate(model, tok, lines, torch.device("cpu"), batch_size=2, max_len=10)
    assert len(hyps) == 3 and all(isinstance(h, str) for h in hyps)
    # Same output whatever the batching (sorting by length must not shuffle results).
    assert hyps == translate(model, tok, lines, torch.device("cpu"), batch_size=1, max_len=10)


def test_bleu_is_scored_against_raw_references(tok):
    model = _model(tok.vocab_size)
    refs = ["the cat eats the fish.", "the dog eats the meat."]
    res = evaluate_bleu(model, tok, FR[:2], refs, torch.device("cpu"), max_len=10)
    assert res["references"] is refs
    assert 0.0 <= res["bleu"] <= 100.0 and "BLEU" in res["bleu_signature"]


def test_fingerprint_detects_a_different_vocab(tok, tmp_path):
    saved = tok.fingerprint()
    check_fingerprint(saved, tok)  # same file: fine

    other = BPETokenizer()
    other.train({"fr": FR[:3] * 5, "en": EN[:3] * 5}, vocab_size=400)
    path = tmp_path / "other.json"
    other.save(str(path))
    with pytest.raises(ValueError, match="does not match"):
        check_fingerprint(saved, JointBPETokenizer(str(path), vocab_threshold=1))
    with pytest.raises(ValueError, match="does not match"):  # same vocab, other threshold
        check_fingerprint(saved, JointBPETokenizer(tok.path, vocab_threshold=5))
    with pytest.raises(ValueError, match="no tokenizer fingerprint"):
        check_fingerprint(None, tok)
