import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

FR = ["le chat mange le poisson", "le chien mange la viande", "les chats dorment"] * 20
EN = ["the cat eats the fish", "the dog eats the meat", "the cats sleep"] * 20


def _trained(vocab_size=60):
    bpe = BPETokenizer()
    bpe.train({"fr": FR, "en": EN}, vocab_size=vocab_size)
    return bpe


def test_joint_vocab_covers_both_languages():
    bpe = _trained()
    assert "le</w>" in bpe.vocab and "the</w>" in bpe.vocab
    assert set(bpe.lang_counts) == {"fr", "en"}
    assert "the</w>" not in bpe.lang_counts["fr"]


def test_ids_are_contiguous():
    bpe = _trained(vocab_size=200)  # runs out of pairs before 200
    assert sorted(bpe.vocab.values()) == list(range(len(bpe.vocab)))
    assert bpe.vocab["<pad>"] == 0 and bpe.vocab["<unk>"] == 1


def test_roundtrip_and_no_unk_on_known_chars():
    bpe = _trained()
    for text in ["le chat dort", "the dog sleeps", "chien poisson cat"]:
        ids = bpe.encode(text)
        assert bpe.unk_id not in ids
        assert bpe.decode(ids) == text


def test_encode_matches_training_segmentation():
    bpe = _trained()
    # Frequent training words should come out as single tokens.
    assert bpe.tokenize("the mange") == ["the</w>", "mange</w>"]


def test_vocab_threshold_splits_other_language_tokens():
    bpe = _trained()
    # "the" never occurs in French, so in French it must be split back down.
    assert bpe.tokenize("the", lang="fr", threshold=1) != ["the</w>"]
    assert "".join(bpe.tokenize("the", lang="fr", threshold=1)) == "the</w>"
    # ...but stays whole in English.
    assert bpe.tokenize("the", lang="en", threshold=1) == ["the</w>"]


def test_add_special():
    bpe = _trained()
    ids = bpe.encode("le chat", add_special=True)
    assert ids[0] == bpe.bos_id and ids[-1] == bpe.eos_id


def test_save_load_roundtrip(tmp_path):
    bpe = _trained()
    path = tmp_path / "bpe.json"
    bpe.save(str(path))
    loaded = BPETokenizer().load(str(path))
    for text, lang in [("le chat mange", "fr"), ("the cats eat", "en")]:
        assert loaded.encode(text) == bpe.encode(text)
        assert loaded.encode(text, lang=lang, threshold=5) == bpe.encode(text, lang=lang, threshold=5)
