# Small Transformer, trained once per tokenizer: which 32k tokenizer translates better?
# Colab cells 8-12, to paste UNDER cells 1-7 of colab.py (same notebook).
#
# Before Cell 8: Runtime -> Change runtime type -> T4 GPU. That starts a fresh
# machine, so Cell 8 restores everything from Drive; cells 1-7 don't need to run again.
#
# What it does:
#   configs/exp/10_small_wmt14_own.yaml vs 10_small_wmt14_spm.yaml: the same small
#   Transformer (d_model 256, 4+4 layers, 15.6M parameters) on the same 1M WMT14
#   pairs, only the tokenizer differs. 2 seeds each, interleaved (own s1, spm s1,
#   own s2, spm s2), so stopping early still leaves a fair comparison.
#   Score: averaged last 5 checkpoints, beam 4, newstest2014 (3,003 sentences),
#   sacreBLEU + chrF, no output-vocab restriction for either.
#
# Everything goes to Drive: MyDrive/transformer-fr-en/wmt14_1m_32k_cpp/small_transformer/
#   <tok>_seed<N>/  checkpoints, last.pt (resume point), train.log, train_summary.json,
#                   test.hyp.en, test_beam4.json
#   results.md / results.json         BLEU / chrF per run (Cell 11)
#   comparison.md / comparison.json   every metric, side by side, with significance (Cell 12)
# Colab disconnected? Run Cell 8, then Cell 10 again: each run continues from its
# last.pt (saved every 1,000 steps) with the same weights, optimizer, LR schedule,
# RNG and position in the epoch; finished runs are skipped.


# %% Cell 8: GPU setup + restore data and both tokenizers from Drive  (~2 min)
import os, subprocess, shutil, json
from pathlib import Path
from google.colab import drive

os.environ["PYTHONUNBUFFERED"] = "1"

def sh(cmd):
    """Run a bash command, print its output live, stop the cell if it fails."""
    p = subprocess.Popen(f"set -o pipefail; {cmd}", shell=True, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"failed (exit {p.returncode}): {cmd}")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU: Runtime -> Change runtime type -> T4 GPU, then run this cell again")
print("GPU:", torch.cuda.get_device_name(0))

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
OUT = Path("/content/drive/MyDrive/transformer-fr-en/wmt14_1m_32k_cpp")
RUNS = OUT / "small_transformer"
RUNS.mkdir(parents=True, exist_ok=True)

REPO = Path("/content/repo")
if not REPO.exists():
    sh(f"git clone -b paper-baseline https://github.com/HoussemBouzamoucha/transformer-fr-en-scratch.git {REPO}")
os.chdir(REPO)
sh("git pull -q && git log --oneline -1")
sh('pip install -q sacrebleu sentencepiece "protobuf>=5"')

CLEAN = Path("data/clean/wmt14_1m")
if not (CLEAN / "train.fr").exists():
    shutil.copy(OUT / "wmt14_clean_1m.tar.gz", "/tmp/clean.tar.gz")
    sh("tar -xzf /tmp/clean.tar.gz")
Path("data/vocab").mkdir(parents=True, exist_ok=True)
for f in ("own_bpe_32k_wmt1m_t10.json", "m2m100_spmbpe_32k.model", "m2m100_spmbpe_32k.vocab"):
    if not Path("data/vocab", f).exists():
        shutil.copy(OUT / f, Path("data/vocab", f))
sh(f"wc -l {CLEAN}/train.fr {CLEAN}/val.fr {CLEAN}/test.fr; ls -la data/vocab")
# Builds the C++ BPE encoder now (the dataset uses it) instead of inside the first run.
sh("python -c \"from src.tokenizer.fast_bpe import _load; print('C++ BPE encoder:', 'ok' if _load() else 'unavailable')\"")

TOKS = {"own": "configs/exp/10_small_wmt14_own.yaml", "spm": "configs/exp/10_small_wmt14_spm.yaml"}


# %% Cell 9 (optional): timing test, 300 steps of our BPE  (~5 min)
# Prints steps/s, then how long Cell 10 takes for the step budget below. Pick
# STEPS in Cell 10 from it (Colab's free tier gives a few GPU hours per day; the
# runs resume, so they can also be spread over several days).
sh(f"python src/training/train.py {TOKS['own']} --seed 1 --max-steps 300 --checkpoint-dir /tmp/timing"
   f" 2>&1 | grep -E 'pairs loaded|Parameters|Batching|Training loop'")
s = json.loads(Path("/tmp/timing/train_summary.json").read_text())
for steps in (10_000, 20_000, 30_000):
    hours = steps / s["steps_per_second"] / 3600
    print(f"  {steps:,} steps: {hours:.1f} h per run, {4 * hours:.1f} h for all 4 runs (+ ~5 min/run to load data)")
shutil.rmtree("/tmp/timing", ignore_errors=True)


# %% Cell 10: train 2 seeds x 2 tokenizers  (hours; re-run after a disconnect to resume)
STEPS = 30_000      # same for every run; ~7 epochs of the 1M pairs at 8,192 tokens/step
SEEDS = [1, 2]

for seed in SEEDS:
    for tok, cfg in TOKS.items():
        run = RUNS / f"{tok}_seed{seed}"
        summary = run / "train_summary.json"
        if summary.exists() and json.loads(summary.read_text())["steps"] >= STEPS:
            print(f"== {run.name}: done, skipping")
            continue
        run.mkdir(parents=True, exist_ok=True)
        print(f"== {run.name}: training to {STEPS:,} steps (resumes from last.pt if there is one)")
        sh(f"python src/training/train.py {cfg} --seed {seed} --max-steps {STEPS}"
           f" --checkpoint-dir '{run}' --resume 2>&1 | tee -a '{run}/train.log'"
           " | grep --line-buffered -E 'pairs loaded|Parameters|Resumed|No .*last.pt|step .*val loss|Averaged|Training loop|Error|Traceback'")


# %% Cell 11: score every finished run on newstest2014 + results table  (~5 min per run)
import statistics

rows = []
for seed in SEEDS:
    for tok, cfg in TOKS.items():
        run = RUNS / f"{tok}_seed{seed}"
        if not (run / "averaged.pt").exists():
            print(f"{run.name}: not finished yet, skipped")
            continue
        scores = run / "test_beam4.json"
        if not scores.exists():
            sh(f"python evaluate.py --config {cfg} --checkpoint '{run}/averaged.pt' --split test"
               f" --beam 4 --no-restrict --examples 3 --json '{scores}'")
        r = json.loads(scores.read_text())
        t = json.loads((run / "train_summary.json").read_text())
        rows.append({"tokenizer": tok, "seed": seed, "bleu": r["bleu"], "chrf": r["chrf"],
                     "best_val_loss": t.get("best_val_loss"), "steps": t["steps"],
                     "steps_per_second": t["steps_per_second"], "signature": r["signature"]})

names = {"own": "our joint BPE", "spm": "SentencePiece BPE"}
lines = ["| tokenizer | seed | test BLEU | chrF | best val loss | steps | steps/s |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {names[r['tokenizer']]} | {r['seed']} | {r['bleu']:.2f} | {r['chrf']:.2f} | "
                 f"{r['best_val_loss']:.4f} | {r['steps']:,} | {r['steps_per_second']:.2f} |")
lines += ["", "| tokenizer | runs | test BLEU mean ± sd | chrF mean ± sd |", "|---|---|---|---|"]
for tok in TOKS:
    b = [r["bleu"] for r in rows if r["tokenizer"] == tok]
    c = [r["chrf"] for r in rows if r["tokenizer"] == tok]
    if b:
        sd = lambda x: statistics.stdev(x) if len(x) > 1 else float("nan")
        lines.append(f"| {names[tok]} | {len(b)} | {statistics.mean(b):.2f} ± {sd(b):.2f} | "
                     f"{statistics.mean(c):.2f} ± {sd(c):.2f} |")
if rows:
    lines += ["", f"sacreBLEU: {rows[0]['signature']}",
              "newstest2014, averaged last 5 checkpoints, beam 4, length penalty 0.6, no output restriction."]
table = "\n".join(lines)
print(table)
(RUNS / "results.md").write_text(table + "\n", encoding="utf-8")
(RUNS / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nsaved {RUNS / 'results.md'} and results.json")


# %% Cell 12: own BPE vs SentencePiece on every metric  (~2 min, after Cell 11)
# Uses the translations Cell 11 saved (test.hyp.en); nothing is retrained or re-decoded.
#   Translation quality  BLEU, chrF, chrF++, TER (lower is better), output/reference length
#   By sentence length   BLEU on short / medium / long source sentences
#   Output problems      unknown tokens in the output, looping repeats, empty outputs
#   Training             val loss per English WORD (per-token losses aren't comparable
#                        across tokenizers, so it's converted with each one's token count),
#                        speed, time to the best checkpoint
#   Significance         paired bootstrap on BLEU (1,000 resamples) for each seed pair:
#                        95% interval of the difference and a p-value
import statistics
import numpy as np
from collections import Counter
from sacrebleu.metrics import BLEU, CHRF, TER
from src.tokenizer.loader import load_tokenizer
from src.utils import load_config

src = [l.strip() for l in open(CLEAN / "test.fr", encoding="utf-8")]
refs = [l.strip() for l in open(CLEAN / "test.en", encoding="utf-8")]
val_fr = [l.strip() for l in open(CLEAN / "val.fr", encoding="utf-8")]
val_en = [l.strip() for l in open(CLEAN / "val.en", encoding="utf-8")]
BUCKETS = [("short (1-15 words)", 1, 15), ("medium (16-30)", 16, 30), ("long (31+)", 31, 10**9)]
src_len = [len(s.split()) for s in src]

def val_tokens_and_words(cfg):
    """Target tokens the val loss was averaged over (pairs kept by max_tokens, </s> included)
    and the English words in those same pairs."""
    c = load_config(cfg)
    tk = load_tokenizer(c)
    fr, en = tk.encode_batch(val_fr, lang="fr"), tk.encode_batch(val_en, lang="en")
    keep = [i for i in range(len(en)) if len(fr[i]) + 2 <= c["data"]["max_tokens"]
            and len(en[i]) + 2 <= c["data"]["max_tokens"]]
    return sum(len(en[i]) + 1 for i in keep), sum(len(val_en[i].split()) for i in keep)

VAL = {tok: val_tokens_and_words(cfg) for tok, cfg in TOKS.items()}

def repeats(h):  # a word 4-gram 3+ times in one output = the decoder looping
    w = h.split()
    return any(c >= 3 for c in Counter(zip(w, w[1:], w[2:], w[3:])).values())

def measure(tok, seed):
    run = RUNS / f"{tok}_seed{seed}"
    if not (run / "test.hyp.en").exists():
        return None
    hyps = (run / "test.hyp.en").read_text(encoding="utf-8").split("\n")[:len(refs)]
    t = json.loads((run / "train_summary.json").read_text())
    bleu = BLEU().corpus_score(hyps, [refs])
    m = {"bleu": bleu.score, "chrf": CHRF().corpus_score(hyps, [refs]).score,
         "chrf_pp": CHRF(word_order=2).corpus_score(hyps, [refs]).score,
         "ter": TER().corpus_score(hyps, [refs]).score,
         "len_ratio": bleu.sys_len / bleu.ref_len}
    for name, lo, hi in BUCKETS:
        idx = [i for i, n in enumerate(src_len) if lo <= n <= hi]
        m[name] = BLEU().corpus_score([hyps[i] for i in idx], [[refs[i] for i in idx]]).score
    words = sum(len(h.split()) for h in hyps) or 1
    m["unk_out"] = 1000 * sum(h.count("<unk>") + h.count("⁇") for h in hyps) / words
    m["repeat"] = 100 * sum(map(repeats, hyps)) / len(hyps)
    m["empty"] = sum(not h.strip() for h in hyps)
    tokens, vwords = VAL[tok]
    m["val_per_word"] = t["best_val_loss"] * tokens / vwords
    m["steps_per_s"] = t["steps_per_second"]
    m["train_h"] = t["train_seconds"] / 3600
    m["best_step"] = t.get("best_step")
    m["_hyps"] = hyps
    return m

M = {(tok, seed): measure(tok, seed) for seed in SEEDS for tok in TOKS}
M = {k: v for k, v in M.items() if v}
done = sorted({s for (_, s) in M if (("own", s) in M and ("spm", s) in M)})
if not done:
    raise RuntimeError("No seed has finished runs for both tokenizers yet (run Cells 10 and 11 first)")

ROWS = [  # key, label, format, better ("high" / "low" / None)
    ("Translation quality (newstest2014)", None, None, None),
    ("bleu", "BLEU", "{:.2f}", "high"),
    ("chrf", "chrF", "{:.2f}", "high"),
    ("chrf_pp", "chrF++", "{:.2f}", "high"),
    ("ter", "TER (lower is better)", "{:.2f}", "low"),
    ("len_ratio", "output / reference length", "{:.3f}", None),
    ("BLEU by source length", None, None, None),
    *[(name, name, "{:.2f}", "high") for name, _, _ in BUCKETS],
    ("Output problems", None, None, None),
    ("unk_out", "unknown tokens per 1k output words", "{:.2f}", "low"),
    ("repeat", "outputs looping (4-gram 3x) %", "{:.2f}", "low"),
    ("empty", "empty outputs", "{:.0f}", "low"),
    ("Training", None, None, None),
    ("val_per_word", "best val loss per English word (nats)", "{:.4f}", "low"),
    ("steps_per_s", "steps / s", "{:.2f}", "high"),
    ("train_h", "training time (h)", "{:.2f}", "low"),
    ("best_step", "step of best val loss", "{:,.0f}", None),
]

def mean_of(tok, key):
    return statistics.mean(M[(tok, s)][key] for s in done)

def better_of(a, b, better):
    if better is None or a == b:
        return "" if better is None else "tie"
    return "own BPE" if (a > b) == (better == "high") else "SentencePiece"

seeds_txt = ", ".join(map(str, done))
out = [f"# Own BPE vs SentencePiece: small Transformer, WMT14 FR->EN (seeds {seeds_txt})", "",
       "| metric | own BPE | SentencePiece | own − SPM | better | per seed (own / SPM) |",
       "|---|---|---|---|---|---|"]
wins = Counter()
for key, label, fmt, better in ROWS:
    if label is None:
        out.append(f"| **{key}** | | | | | |")
        continue
    a, b = mean_of("own", key), mean_of("spm", key)
    w = better_of(a, b, better)
    if w in ("own BPE", "SentencePiece"):
        wins[w] += 1
    per_seed = "; ".join(f"s{s}: {fmt.format(M[('own', s)][key])} / {fmt.format(M[('spm', s)][key])}" for s in done)
    out.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {a - b:+.3f} | {w} | {per_seed} |")

# Paired bootstrap on BLEU, own vs SPM of the same seed (same 3,003 sentences).
bleu = BLEU()
out += ["", "## Is the BLEU difference real? (paired bootstrap, 1,000 resamples)", "",
        "| seed | own BLEU | SPM BLEU | own − SPM | 95% interval | p-value |", "|---|---|---|---|---|---|"]
rng = np.random.default_rng(12345)
significant = []
for s in done:
    ho, hs = M[("own", s)]["_hyps"], M[("spm", s)]["_hyps"]
    try:  # sentence-level BLEU statistics, then resample sums (how sacreBLEU's own test works)
        so = np.array(bleu._extract_corpus_statistics(ho, [refs]))
        ss = np.array(bleu._extract_corpus_statistics(hs, [refs]))
        idx = rng.integers(0, len(refs), (1000, len(refs)))
        diffs = np.array([bleu._compute_score_from_stats(so[i].sum(0).tolist()).score
                          - bleu._compute_score_from_stats(ss[i].sum(0).tolist()).score for i in idx])
    except AttributeError:  # older/newer sacreBLEU without those helpers: slower, 200 resamples
        diffs = []
        for _ in range(200):
            i = rng.integers(0, len(refs), len(refs))
            diffs.append(bleu.corpus_score([ho[k] for k in i], [[refs[k] for k in i]]).score
                         - bleu.corpus_score([hs[k] for k in i], [[refs[k] for k in i]]).score)
        diffs = np.array(diffs)
    d = M[("own", s)]["bleu"] - M[("spm", s)]["bleu"]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1) if d > 0 else (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)
    significant.append(p < 0.05)
    out.append(f"| {s} | {M[('own', s)]['bleu']:.2f} | {M[('spm', s)]['bleu']:.2f} | {d:+.2f} | "
               f"[{lo:+.2f}, {hi:+.2f}] | {p:.3f} |")

mean_d = statistics.mean(M[("own", s)]["bleu"] - M[("spm", s)]["bleu"] for s in done)
leader = "own BPE" if mean_d > 0 else "SentencePiece"
if all(significant) and len({np.sign(M[('own', s)]['bleu'] - M[('spm', s)]['bleu']) for s in done}) == 1:
    verdict = f"{leader} is better: +{abs(mean_d):.2f} BLEU on average, significant (p < 0.05) for every seed."
elif any(significant):
    verdict = (f"{leader} leads by {abs(mean_d):.2f} BLEU on average, but it is not significant for every "
               "seed: a small or seed-dependent difference.")
else:
    verdict = (f"No significant difference ({leader} +{abs(mean_d):.2f} BLEU on average, within noise). "
               "Tie-breakers: <unk> rate and encoding speed (see report.txt).")
out += ["", f"Metrics won (means over seeds): {dict(wins)}", "", f"**Verdict:** {verdict}", "",
        "newstest2014 (3,003 sentences), averaged last 5 checkpoints, beam 4, length penalty 0.6, "
        "no output restriction; sacreBLEU defaults (13a)."]

table = "\n".join(out)
print(table)
(RUNS / "comparison.md").write_text(table + "\n", encoding="utf-8")
(RUNS / "comparison.json").write_text(json.dumps(
    {f"{tok}_seed{s}": {k: v for k, v in m.items() if k != "_hyps"} for (tok, s), m in M.items()}
    | {"verdict": verdict, "seeds_compared": done}, indent=2), encoding="utf-8")
print(f"\nsaved {RUNS / 'comparison.md'} and comparison.json")
drive.flush_and_unmount()   # make sure everything is on Drive; Cell 8 mounts it again
