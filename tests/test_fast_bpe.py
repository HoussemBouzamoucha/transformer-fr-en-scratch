"""The C++ batch encoder (src/tokenizer/fast_bpe.cpp) must give exactly the ids of
the Python reference, BPETokenizer.encode(). Skipped when no C++ compiler exists."""
import random

import pytest

from src.tokenizer import fast_bpe
from src.tokenizer.bpe import BPETokenizer

if fast_bpe._load() is None:
    pytest.skip(f"C++ encoder unavailable: {fast_bpe._LIB_ERROR}", allow_module_level=True)

FR = [
    "L'homme regarde par la fenêtre.", "Qu'il vienne, s'il te plaît !", "C'est l'été ; on part à 8h30...",
    "« Bonjour », dit-elle : « ça va ? »", "Un chien noir court dans l'herbe verte.",
    "Les enfants jouent au football près de l'école.", "aujourd'hui, j'ai vu 3 oiseaux",
] * 20
EN = [
    "The man's dog doesn't bark.", "Two women aren't sitting on a bench.", "It's 5 o'clock -- let's go!",
    "A black dog runs through the green grass.", "Children play football near the school.",
    "\"Hello,\" she said: \"how's it going?\"", "rock'n'roll isn't dead",
] * 20

# Every rule normalize() / _pieces() has, plus inputs the training text never had.
TRICKY = [
    "", " ", "   \t ", "l’homme", "“quoted” and «guillemets»", "café crème",
    "été", "́leading mark", "zero​width­soft", "nbsp ; ! ?",
    "ctrl\x00\x07char\x0bs\x0c\x1f\x7f", "tab\tand\rcarriage", "line\nbreak inside", "a\n\nb",
    "glue⁐mark ⁐ ⁐⁐", "l'' ''x '' 'a' b'", "qu'aujourd'hui l'l'x", "x'", "'x", "'",
    "emoji \U0001F600 ok", "中文 Αβ", "under_score __init__", "…–−´`",
    "numbers 3,14 1.000.000 2e10 ½ ²", "　full　width　", " sep ", "\x1c\x1d\x1e",
    "MiXeD CaSe WORDS", "punct!!!???...,,,;;;", "a.b.c-d/e\\f", "\ud800 lone surrogate",
]


def random_texts(n, seed=0):
    rng = random.Random(seed)
    alphabet = ("abcdefghijklmnopqrstuvwxyzéèêàçùôœ'’.,;:!?-« »\"()0123456789  ⁐́\n\t"
                "ABCÉÈ_​\x07中\U0001F600")
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(n)]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train({"fr": FR, "en": EN}, vocab_size=300, fit_threshold=2)
    return t


def check(t, texts, lang, threshold=0, **kw):
    ref = [t.encode(x, lang=lang, threshold=threshold, **kw) for x in texts]
    got = t.encode_batch(texts, lang=lang, threshold=threshold, **kw)
    bad = [(x, a, b) for x, a, b in zip(texts, ref, got) if a != b]
    assert not bad, f"{len(bad)} mismatches, first: {bad[0]!r}"


@pytest.mark.parametrize("lang", ["fr", "en", None, "de"])
@pytest.mark.parametrize("threshold", [0, 1, 2, 5])
def test_same_ids_as_python(tok, lang, threshold):
    texts = FR + EN + TRICKY + random_texts(500)
    check(tok, texts, lang, threshold)


def test_unknown_characters_are_unk(tok):
    # "中" "⁐😀" (emoji is punctuation to \w): both characters unseen -> <unk>, </w> after each piece
    ids = tok.encode_batch(["中\U0001F600"], lang="fr")[0]
    eow, glue, unk = tok.vocab[tok.EOW], tok.vocab[tok.GLUE], tok.unk_id
    assert ids == [unk, eow, glue, unk, eow] == tok.encode("中\U0001F600", lang="fr")


def test_add_special_and_empty(tok):
    assert tok.encode_batch([]) == []
    check(tok, ["", "a", "l'homme"], "fr", add_special=True)


def test_threads_and_repeated_calls_agree(tok):
    texts = random_texts(3000, seed=1)
    one = tok.encode_batch(texts, lang="fr", threshold=2, num_threads=1)
    assert tok.encode_batch(texts, lang="fr", threshold=2, num_threads=8) == one
    assert tok.encode_batch(texts, lang="fr", threshold=2, num_threads=8) == one  # warm caches
    check(tok, texts, "fr", 2)


def test_without_pretokenize_and_normalize():
    t = BPETokenizer(pretokenize=0, normalize=False)
    t.train({"fr": FR, "en": EN}, vocab_size=250)
    check(t, FR + EN + TRICKY + random_texts(300, seed=2), "fr", 1)


def test_retrain_rebuilds_tables(tok, tmp_path):
    t = BPETokenizer()
    t.train({"fr": FR[:30], "en": EN[:30]}, vocab_size=120)
    t.encode_batch(FR[:5], lang="fr")  # builds the C++ tables
    t.train({"fr": FR, "en": EN}, vocab_size=300, fit_threshold=2)
    check(t, FR + TRICKY, "fr", 2)
    path = tmp_path / "v.json"
    t.save(str(path))
    check(BPETokenizer().load(str(path)), FR + EN + TRICKY, "en", 2)


# ------------------------------------------------------------------ training
def trained_bytes(tmp_path, name, corpora, init=None, python=False, **kw):
    t = BPETokenizer(**(init or {}))
    (t._train_python if python else t.train)(corpora, **kw)
    path = tmp_path / f"{name}_{'py' if python else 'cpp'}.json"
    t.save(str(path))
    return path.read_bytes(), t


def small_alphabet(n, seed):
    # Few letters, long words: many equal pair counts, so the tie-break (smallest
    # pair by code points) decides most merges.
    rng = random.Random(seed)
    return ["".join(rng.choice("abc ab'.") for _ in range(rng.randint(1, 40))) for _ in range(n)]


# A lone surrogate can't be saved as UTF-8 JSON (by either trainer): encoding tests only.
TRAIN_TRICKY = [x for x in TRICKY if "\ud800" not in x]


TRAIN_CASES = {
    "fit10": (dict(vocab_size=300, fit_threshold=10), None),
    "fit2": (dict(vocab_size=300, fit_threshold=2), None),
    "fit0": (dict(vocab_size=200, fit_threshold=0), None),
    "classic": (dict(vocab_size=250), None),
    "rare_chars": (dict(vocab_size=250, fit_threshold=2, min_char_count=30), None),
    "raw": (dict(vocab_size=250, fit_threshold=1), dict(pretokenize=0, normalize=False)),
    "too_big": (dict(vocab_size=5000, fit_threshold=2), None),  # runs out of merges
}


@pytest.mark.parametrize("case", list(TRAIN_CASES))
def test_training_matches_python(case, tmp_path):
    kw, init = TRAIN_CASES[case]
    corpora = {"fr": FR + TRAIN_TRICKY + random_texts(400, 3), "en": EN + random_texts(400, 4)}
    cpp, _ = trained_bytes(tmp_path, case, corpora, init, **kw)
    py, _ = trained_bytes(tmp_path, case, corpora, init, python=True, **kw)
    assert cpp == py


@pytest.mark.parametrize("fit", [None, 0, 3])
def test_training_ties(fit, tmp_path):
    corpora = {"fr": small_alphabet(600, 5), "en": small_alphabet(600, 6)}
    cpp, _ = trained_bytes(tmp_path, "ties", corpora, vocab_size=400, fit_threshold=fit)
    py, _ = trained_bytes(tmp_path, "ties", corpora, vocab_size=400, fit_threshold=fit, python=True)
    assert cpp == py


def test_training_plain_list_and_empty(tmp_path):
    lines = FR[:50] + EN[:50]
    assert trained_bytes(tmp_path, "list", lines, vocab_size=150)[0] == \
        trained_bytes(tmp_path, "list", lines, vocab_size=150, python=True)[0]
    empty = {"fr": [], "en": [""]}
    assert trained_bytes(tmp_path, "empty", empty, vocab_size=150, fit_threshold=2)[0] == \
        trained_bytes(tmp_path, "empty", empty, vocab_size=150, fit_threshold=2, python=True)[0]


def test_trained_tokenizer_encodes_like_python(tmp_path):
    _, t = trained_bytes(tmp_path, "enc", {"fr": FR, "en": EN}, vocab_size=300, fit_threshold=2)
    check(t, FR + EN + TRICKY, "fr", 2)


def test_fallback_without_cpp(tok, monkeypatch):
    monkeypatch.setenv("BPE_FAST", "0")
    t = BPETokenizer()
    t.train({"fr": FR, "en": EN}, vocab_size=200)
    assert t.encode_batch(FR[:3], lang="fr") == [t.encode(x, lang="fr") for x in FR[:3]]
    assert t._fast is False
