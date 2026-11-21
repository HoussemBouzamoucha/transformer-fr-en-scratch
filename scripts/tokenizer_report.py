"""Our joint BPE vs a SentencePiece model, side by side on what matters to the Transformer.

    python scripts/tokenizer_report.py --data-dir data/clean/wmt14_1m \
        --own data/vocab/own_bpe_32k_wmt1m_t10.json --threshold 10 \
        --spm data/vocab/m2m100_spmbpe_32k.model --json report.json

Every number is measured the way training/eval will see it:
  - the model is --exp (default configs/exp/00_paper_base.yaml), built with the
    real build_model; pairs longer than its data.max_tokens are dropped like dataset.py
  - batches come from the real TokenBatchSampler (batch_tokens x accum_steps)
  - our BPE encodes per language with --threshold, via the multithreaded C++
    batch encoder (same ids as the Python encode(); tests/test_fast_bpe.py)
  - SPM gets the same normalize() our BPE applies: its training data (data/clean)
    is normalized, but val/test are raw, so without it « » “ ” ’ – become <unk>
Data: --sample training pairs for length/cost/vocab statistics; val + test (raw)
for <unk>, word splitting and round trips. "better" marks a winner only when the
gap is over 1%.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import TokenBatchSampler
from src.model.transformer import build_model
from src.tokenizer.loader import JointBPETokenizer, SentencePieceTokenizer
from src.tokenizer.normalize import normalize
from src.tokenizer.sampling import sample_pairs
from src.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True, help="cleaned data: train/val/test .fr/.en")
ap.add_argument("--own", required=True, help="our BPE vocab .json")
ap.add_argument("--threshold", type=int, default=10, help="tokenizer.vocab_threshold for our BPE")
ap.add_argument("--spm", required=True, help="SentencePiece .model")
ap.add_argument("--exp", default="configs/exp/00_paper_base.yaml", help="model/training config")
ap.add_argument("--sample", type=int, default=100_000, help="training pairs to measure on")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--corpus-pairs", type=int, default=None, help="default: lines in train.fr")
ap.add_argument("--json", default=None, help="also write every number here")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

data = Path(args.data_dir)
EXP = load_config(args.exp)
D_MODEL, MAX_TOKENS = EXP["model"]["d_model"], EXP["data"]["max_tokens"]
BATCH, ACCUM = EXP["training"]["batch_tokens"], EXP["training"].get("accum_steps", 1)
if args.corpus_pairs is None:
    with open(data / "train.fr", "rb") as f:
        args.corpus_pairs = sum(1 for _ in f)
CORPUS = args.corpus_pairs
print(f"{os.cpu_count()} CPU cores | {args.exp}: d_model={D_MODEL}, max_tokens={MAX_TOKENS}, "
      f"batch {BATCH} x {ACCUM} tokens | corpus {CORPUS:,} pairs")


def load_toks():
    """Fresh tokenizers (empty caches), all with encode / encode_batch / decode / pieces."""
    spm_tok = SentencePieceTokenizer(args.spm)
    spm_tok.encode = lambda text, lang=None: spm_tok.sp.encode(normalize(text))
    spm_tok.encode_batch = lambda texts, lang=None: spm_tok.sp.encode(
        [normalize(x) for x in texts], num_threads=-1)
    spm_tok.pieces = lambda text, lang=None: spm_tok.sp.encode(normalize(text), out_type=str)
    own = JointBPETokenizer(args.own, vocab_threshold=args.threshold)
    own.pieces = lambda text, lang=None: own.bpe.tokenize(text, lang=lang, threshold=own.threshold)
    return {"own BPE": own, "SPM BPE": spm_tok}


toks = load_toks()
NAMES = list(toks)
toks["own BPE"].encode_batch(["warm up"], lang="fr")  # compiles the C++ encoder once (~10 s)
if toks["own BPE"].bpe._fast is False:
    print("WARNING: C++ encoder unavailable, our BPE runs in Python (same ids, slower)")


def read(split):
    return {l: [x.strip() for x in open(data / f"{split}.{l}", encoding="utf-8")] for l in ("fr", "en")}


train, _ = sample_pairs(data, args.sample, args.seed)
val = read("val")
held = {l: val[l] + read("test")[l] for l in ("fr", "en")}
words = {l: [w for x in held[l] for w in x.split()] for l in ("fr", "en")}
print(f"{len(train['fr']):,} training pairs, {len(held['fr']):,} val+test pairs")


def renyi_efficiency(counts, vocab_size, alpha=2.5):
    """Zouhar et al. 2023: Renyi entropy of the token distribution / log2 V; in their
    experiments it predicted BLEU better than sequence length did."""
    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    return (math.log2((p ** alpha).sum()) / (1 - alpha)) / math.log2(vocab_size)


R = {}
for name, t in toks.items():
    r = R[name] = {}
    V = t.vocab_size
    enc = {l: t.encode_batch(train[l], lang=l) for l in ("fr", "en")}
    L = {l: np.array([len(x) for x in enc[l]]) for l in ("fr", "en")}
    W = {l: np.array([len(x.split()) for x in train[l]]) for l in ("fr", "en")}
    C = {l: sum(len(x) for x in train[l]) for l in ("fr", "en")}

    # model
    model = build_model(EXP["model"], V, t.pad_id)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    r["vocab"] = V
    r["x64"] = "yes" if V % 64 == 0 else "no"
    r["params"] = n_params / 1e6
    r["emb_share"] = 100 * V * D_MODEL / n_params
    tgt_ids = t.target_token_ids("en")
    r["softmax"] = len(tgt_ids) if tgt_ids else V

    # sequence length (no <s> </s>)
    for l in ("fr", "en"):
        r[f"{l}_mean"] = float(L[l].mean())
        r[f"{l}_p95"] = float(np.percentile(L[l], 95))
        r[f"{l}_p99"] = float(np.percentile(L[l], 99))
        r[f"{l}_max"] = int(L[l].max())
        r[f"{l}_fert"] = float(L[l].sum() / W[l].sum())
        r[f"{l}_cpt"] = float(C[l] / L[l].sum())
    r["ratio"] = float(L["en"].sum() / L["fr"].sum())
    keep = (L["fr"] + 2 <= MAX_TOKENS) & (L["en"] + 2 <= MAX_TOKENS)  # dataset.py adds <s> </s>
    r["dropped"] = float(100 * (1 - keep.mean()))

    # training cost, on the pairs the dataset keeps, with <s> </s> like training
    S, T = L["fr"][keep] + 2, L["en"][keep] + 2
    scale = CORPUS / keep.sum()
    r["tok_epoch"] = float((S.sum() + T.sum()) * scale / 1e6)
    batches = TokenBatchSampler(list(zip(S.tolist(), T.tolist())), BATCH, shuffle=False)._batches()
    real = padded = 0
    for b in batches:
        real += S[b].sum() + T[b].sum()
        padded += len(b) * (S[b].max() + T[b].max())
    r["steps_epoch"] = float(len(batches) * scale / ACCUM)
    r["padding"] = float(100 * (1 - real / padded))
    Sf, Tf = S.astype(np.float64), T.astype(np.float64)
    r["attn"] = float(np.mean(Sf ** 2 + Tf ** 2 + Sf * Tf))  # enc self + dec self + cross

    # vocab use (training sample)
    cnt = {l: Counter(i for x in enc[l] for i in x) for l in ("fr", "en")}
    both = cnt["fr"] + cnt["en"]
    r["used"] = 100 * len(both) / V
    r["rare"] = 100 * sum(1 for i in range(V) if both.get(i, 0) < 100) / V
    r["shared"] = 100 * len(set(cnt["fr"]) & set(cnt["en"])) / max(1, len(both))
    r["renyi"] = renyi_efficiency(both, V)

    # held-out: <unk>, word splitting, round trip
    unk = t.bpe.unk_id if hasattr(t, "bpe") else t.sp.unk_id()
    h_enc = {l: t.encode_batch(held[l], lang=l) for l in ("fr", "en")}
    n_tok = sum(len(x) for l in h_enc for x in h_enc[l])
    r["unk"] = 1000 * sum(x.count(unk) for l in h_enc for x in h_enc[l]) / n_tok
    w_len = np.concatenate([[len(x) for x in t.encode_batch(words[l], lang=l)] for l in ("fr", "en")])
    r["whole"] = float(100 * (w_len == 1).mean())
    r["split3"] = float(100 * (w_len >= 3).mean())
    bad = []
    for x, ids in zip(held["en"], h_enc["en"]):
        want = " ".join(normalize(x).split())
        got = " ".join(t.decode(ids).split())
        if got != want:
            bad.append((want, got))
    r["roundtrip"] = 100 * (1 - len(bad) / len(held["en"]))
    r["_bad"] = bad[:5]

    # speed: cold (fresh tokenizer), the training sample, batch on all cores like the dataset
    fresh = load_toks()[name]
    t0 = time.perf_counter()
    for l in ("fr", "en"):
        fresh.encode_batch(train[l], lang=l)
    r["speed"] = 2 * len(train["fr"]) / (time.perf_counter() - t0)
    r["enc_min"] = 2 * CORPUS / r["speed"] / 60
    print(f"  {name} done")

ROWS = [  # (key, label, format, better: "low" / "high" / None, why it matters); label None = section
    ("Model", None, None, None, None),
    ("vocab", "vocab size", "{:,}", None, "embedding rows + softmax width"),
    ("x64", "vocab multiple of 64", "{}", None, "fp16 tensor-core alignment of the softmax"),
    ("params", "parameters (M)", "{:.1f}", None, "with 3-way tied embeddings"),
    ("emb_share", "embedding share of params %", "{:.1f}", None, "rest is the 6+6 layers"),
    ("softmax", "EN softmax size (restricted)", "{:,}", "low", "decoder output cost per token"),
    ("Sequence length (train sample, no <s></s>)", None, None, None, None),
    ("fr_mean", "FR tokens/sentence mean", "{:.1f}", "low", "encoder length"),
    ("en_mean", "EN tokens/sentence mean", "{:.1f}", "low", "decoder steps, beam search time"),
    ("fr_p95", "FR p95", "{:.0f}", "low", None),
    ("en_p95", "EN p95", "{:.0f}", "low", None),
    ("fr_p99", "FR p99", "{:.0f}", "low", None),
    ("en_p99", "EN p99", "{:.0f}", "low", None),
    ("fr_max", "FR max", "{}", "low", None),
    ("en_max", "EN max", "{}", "low", None),
    ("fr_fert", "FR tokens/word", "{:.3f}", "low", "1.0 = whole words"),
    ("en_fert", "EN tokens/word", "{:.3f}", "low", None),
    ("fr_cpt", "FR chars/token", "{:.2f}", "high", None),
    ("en_cpt", "EN chars/token", "{:.2f}", "high", None),
    ("ratio", "EN/FR token ratio", "{:.3f}", None, "length balance src vs tgt"),
    ("dropped", f"pairs dropped (> {MAX_TOKENS} tokens) %", "{:.2f}", "low", "training data lost"),
    (f"Training cost (per epoch of {CORPUS:,} pairs)", None, None, None, None),
    ("tok_epoch", "tokens per epoch (M, src+tgt)", "{:.1f}", "low", "compute per epoch"),
    ("steps_epoch", "optimizer steps per epoch", "{:,.0f}", "low", "TokenBatchSampler + accum"),
    ("padding", "padding in batches %", "{:.1f}", "low", "wasted compute"),
    ("attn", "attention cost (mean s²+t²+st)", "{:,.0f}", "low", "quadratic in length"),
    ("Vocab use (train sample)", None, None, None, None),
    ("used", "vocab used %", "{:.1f}", "high", "unused rows = dead parameters"),
    ("rare", "tokens seen < 100x %", "{:.1f}", "low", "undertrained embeddings"),
    ("shared", "FR∩EN shared tokens %", "{:.1f}", None, "tied embeddings share these"),
    ("renyi", "Rényi efficiency (α=2.5)", "{:.4f}", "high", "Zouhar 2023: tracks BLEU"),
    ("Held-out (val + test, raw)", None, None, None, None),
    ("unk", "<unk> per 1k tokens", "{:.3f}", "low", "lost information"),
    ("whole", "words kept whole %", "{:.1f}", "high", None),
    ("split3", "words in 3+ pieces %", "{:.1f}", "low", "harder to learn"),
    ("roundtrip", "EN decode(encode(x)) exact %", "{:.2f}", "high", "BLEU scores detokenized text"),
    (f"Speed (cold, batch, {os.cpu_count()} cores)", None, None, None, None),
    ("speed", "sentences/s", "{:,.0f}", "high", None),
    ("enc_min", f"encode {CORPUS:,} pairs (min)", "{:.1f}", "low", "dataset build time"),
]


def winner(key, better):
    a, b = R[NAMES[0]][key], R[NAMES[1]][key]
    if not better or max(abs(a), abs(b)) == 0:
        return ""
    if abs(a - b) / max(abs(a), abs(b)) <= 0.01:
        return "tie"
    return NAMES[0] if (a < b) == (better == "low") else NAMES[1]


w = 38
print(f"\n{'':<{w}}{NAMES[0]:>14}{NAMES[1]:>14}  {'better':<10}why it matters")
score = Counter()
for key, label, fmt, better, why in ROWS:
    if label is None:
        print(f"\n-- {key} " + "-" * max(4, w + 40 - len(key)))
        continue
    win = winner(key, better)
    if win in NAMES:
        score[win] += 1
    a, b = R[NAMES[0]][key], R[NAMES[1]][key]
    print(f"{label:<{w}}{fmt.format(a):>14}{fmt.format(b):>14}  {win:<10}{why or ''}")
print(f"\nmetrics won (>1% gap): {dict(score)}  (not equally important; BLEU decides)")

print("\nEN round-trip failures (first 5 each): normalized input, then decode(encode(input))")
for name in NAMES:
    print(f"  {name}:" + ("  none" if not R[name]["_bad"] else ""))
    for want, got in R[name]["_bad"]:
        print(f"    {want[:110]}\n    {got[:110]}\n")

print("Side by side (val):")
for l, i in (("fr", 0), ("fr", 7), ("en", 0), ("en", 7)):
    x = val[l][i]
    print(f"\n  [{l}] {x}")
    for name, t in toks.items():
        p = t.pieces(x, lang=l)
        print(f"    {name:<8} {len(p):>3}  {' '.join(p)}")

if args.json:
    out = {"args": vars(args), "cpu_count": os.cpu_count(),
           "results": {n: {k: v for k, v in r.items()} for n, r in R.items()},
           "winners": {key: winner(key, better) for key, label, _, better, _ in ROWS if label}}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.json}")
