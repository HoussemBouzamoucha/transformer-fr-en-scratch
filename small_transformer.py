# Small Transformer, trained once per tokenizer: which 32k tokenizer translates better?
# Colab cells 8-11, to paste UNDER cells 1-7 of colab.py (same notebook).
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
#   results.md / results.json   the final table
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
drive.flush_and_unmount()   # make sure everything is on Drive; Cell 8 mounts it again
