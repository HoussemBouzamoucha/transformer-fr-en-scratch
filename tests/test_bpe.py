import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe import BPETokenizer

FR = ["le chat mange le poisson", "le chien mange la viande", "les chats dorment"] * 20
EN = ["the cat eats the fish", "the dog eats the meat", "the cats sleep"] * 20


def _trained(vocab_size=400):
    bpe = BPETokenizer()
    bpe.train({"fr": FR, "en": EN}, vocab_size=vocab_size)
    return bpe


def test_joint_vocab_covers_both_languages():
    bpe = _trained()
    assert "le</w>" in bpe.vocab and "the</w>" in bpe.vocab
    assert set(bpe.lang_counts) == {"fr", "en"}
    assert "the</w>" not in bpe.lang_counts["fr"]


def test_ids_are_contiguous():
    bpe = _trained(vocab_size=1500)  # runs out of pairs before 1500
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


def test_punctuation_is_split_off_words():
    bpe = BPETokenizer()
    bpe.train({"fr": ["le chat mange. le chat dort, le chat"] * 20,
               "en": ["the cat eats. the cat sleeps, the cat"] * 20}, vocab_size=420)
    # "chat" is the same token whether or not punctuation follows it.
    assert bpe.tokenize("chat.")[0] == bpe.tokenize("chat")[0] == "chat</w>"
    assert bpe.tokenize("chat.") == ["chat</w>", bpe.GLUE + ".</w>"]


def test_pieces_one_join_mark_per_boundary():
    bpe, g = BPETokenizer(), BPETokenizer.GLUE
    assert bpe._pieces("fenêtre.") == ["fenêtre", g + "."]
    # Punctuation followed by punctuation: the period is the same token as usual.
    assert bpe._pieces('chose."') == ["chose", g + ".", g + '"']
    assert bpe._pieces('"Hello."') == ['"' + g, "Hello", g + ".", g + '"']
    assert bpe._pieces("c?!") == ["c", g + "?", g + "!"]
    # Punctuation inside a word needs both sides marked.
    assert bpe._pieces("volley-ball") == ["volley", g + "-" + g, "ball"]
    assert bpe._pieces("chat") == ["chat"]


def test_apostrophes_follow_each_language():
    bpe, g = BPETokenizer(), BPETokenizer.GLUE
    # French elision: apostrophe stays with the short word on the left.
    assert bpe._pieces("l'herbe", "fr") == ["l'" + g, "herbe"]
    assert bpe._pieces("qu'il", "fr") == ["qu'" + g, "il"]
    assert bpe._pieces("aujourd'hui", "fr") == ["aujourd'" + g, "hui"]
    # English clitics: apostrophe stays with what follows.
    assert bpe._pieces("man's", "en") == ["man", g + "'s"]
    assert bpe._pieces("don't", "en") == ["don", g + "'t"]
    # Unknown language: standalone.
    assert bpe._pieces("l'herbe") == ["l", g + "'", g + "herbe"] or         bpe._pieces("l'herbe") == ["l", g + "'" + g, "herbe"]


def test_punctuation_roundtrip_is_exact():
    bpe = BPETokenizer()
    bpe.train({"fr": FR + ["l'herbe verte. \"Le chat,\" dit-il... c?! aujourd'hui"] * 5,
               "en": EN + ["the cat's 2 dogs! a -- b don't"] * 5}, vocab_size=420)
    cases = [("l'herbe verte.", "fr"), ('"Le chat," dit-il.', "fr"), ("aujourd'hui", "fr"),
             ("the cat's 2 dogs!", "en"), ("a -- b ... c?!", "en"), ("don't", "en"),
             ("l'herbe", None)]
    for text, lang in cases:
        assert bpe.decode(bpe.encode(text, lang=lang)) == text, (text, lang)


def test_normalization_unifies_typography():
    bpe = BPETokenizer()
    bpe.train({"fr": FR + ["l'herbe"] * 10, "en": EN}, vocab_size=420)
    curly = "l’herbe ?"          # curly apostrophe + non-breaking space
    straight = "l'herbe ?"
    assert bpe.encode(curly, lang="fr") == bpe.encode(straight, lang="fr")
    decomposed = "été"          # é written as e + combining accent
    assert bpe.encode(decomposed) == bpe.encode("été")
    assert bpe.decode(bpe.encode(curly, lang="fr")) == straight


def test_rare_characters_become_unk_but_ascii_is_kept():
    bpe = BPETokenizer()
    bpe.train({"fr": FR + ["chat 中 ☺"], "en": EN + ["cat $"]},
              vocab_size=420, min_char_count=5)
    assert "中" not in bpe.vocab and "☺" not in bpe.vocab   # junk: dropped
    assert "$" in bpe.vocab                                            # ASCII: kept
    assert bpe.unk_id in bpe.encode("中")
    assert bpe.unk_id not in bpe.encode("$ chat")


def test_target_ids_exclude_french_only_tokens_and_cover_targets():
    bpe = _trained()
    allowed = set(bpe.target_token_ids("en", threshold=1))
    assert bpe.vocab["the</w>"] in allowed
    assert bpe.vocab["le</w>"] not in allowed          # never seen in English
    assert bpe.eos_id in allowed and bpe.pad_id not in allowed and bpe.bos_id not in allowed
    # Every token of an English target encoded the same way is allowed.
    for text in EN + ["chien poisson"]:
        assert set(bpe.encode(text, lang="en", threshold=1)) <= allowed, text


def test_old_files_load_without_pretokenize(tmp_path):
    bpe = BPETokenizer(pretokenize=0, normalize=False)
    bpe.train({"fr": FR, "en": EN}, vocab_size=400)
    path = tmp_path / "old.json"
    bpe.save(str(path))
    loaded = BPETokenizer().load(str(path))
    assert loaded.pretokenize == 0 and loaded.normalize is False


def test_retired_pretokenize_v1_files_are_rejected(tmp_path):
    import json
    import pytest
    bpe = _trained()
    path = tmp_path / "v1.json"
    bpe.save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pretokenize"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrain"):
        BPETokenizer().load(str(path))


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
