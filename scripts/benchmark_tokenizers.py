"""Train the same transformer with each tokenizer and compare translation quality.

    python scripts/train_baseline_tokenizers.py                  # once: builds tokenizers + configs
    python scripts/benchmark_tokenizers.py                       # trains what's missing, evaluates all
    python scripts/benchmark_tokenizers.py --seeds 42 43 44      # 3 training seeds per tokenizer

Every config in --configs differs ONLY in its tokenizer (same model, data, seed,
steps). With --seeds, each tokenizer is trained once per seed (the tokenizer itself
is fixed; only model init, dropout and batch order change). The config's own seed
reuses the existing checkpoint folder; other seeds go to <checkpoint_dir>_seed<N>.

Each run is trained in its own process; a run whose checkpoint already exists is
not retrained and its scores are cached next to it, so the script can be re-run
after a Colab disconnect and continues where it stopped. Tokenizers that encode
identically (the retrained M2M-100 and NLLB do) are detected and evaluated once.

Reported per tokenizer:
  - tokens per sentence (test), <unk> count
  - vocabulary utilization on the training set: share of the vocab that is ever
    used, used >= 10 times (enough to learn an embedding), and used in both languages
  - BLEU / chrF: mean and std over seeds
  - training speed: steps/s, target tokens/s, minutes until the best checkpoint
    (from train_summary.json, written by src/training/train.py)

Results: printed tables + logs/tokenizer_benchmark_<split>.md / .json
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.eval.metrics import evaluate_bleu
from src.inference.translator import load_model
from src.tokenizer.loader import load_tokenizer
from src.utils import load_config

N_SPECIAL = 4  # <pad> <unk> <s> </s> = ids 0-3 for every tokenizer here
WELL_TRAINED = 10

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="configs/tokenizers/thin_slice_*.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--seeds", type=int, nargs="*", help="training seeds (default: each config's own seed)")
ap.add_argument("--no-train", action="store_true", help="only evaluate runs that already have a checkpoint")
ap.add_argument("--only", nargs="*", help="config names (e.g. thin_slice_gpt2) to include")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

configs = sorted(glob.glob(args.configs), key=lambda p: (not p.endswith("_ours.yaml"), p))
if args.only:
    configs = [c for c in configs if Path(c).stem in args.only]
if not configs:
    sys.exit(f"No configs match {args.configs}. Run scripts/train_baseline_tokenizers.py first.")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f]


def signature(tok, lines):
    h = hashlib.sha1()
    for lang, ls in lines.items():
        for l in ls:
            h.update(str(tok.encode(l, lang=lang)).encode())
    return h.hexdigest()


def utilization(tok, data_dir):
    """How much of the vocab the training data actually uses."""
    counts = {}
    for lang in ("fr", "en"):
        c = Counter()
        for l in read_lines(data_dir / f"train.{lang}"):
            c.update(tok.encode(l, lang=lang))
        counts[lang] = c
    total = counts["fr"] + counts["en"]
    real = tok.vocab_size - N_SPECIAL
    used = {i for i in total if i >= N_SPECIAL}
    both = {i for i in used if counts["fr"][i] and counts["en"][i]}
    return {
        "util_used": len(used) / real,
        "util_well": sum(1 for i in used if total[i] >= WELL_TRAINED) / real,
        "util_shared": len(both) / real,
        "unused_tokens": real - len(used),
    }


def seed_config(cfg_path, cfg, seed):
    """Config + checkpoint dir for one seed. The config's own seed is the existing run."""
    base_dir = cfg["training"].get("checkpoint_dir", f"checkpoints/{Path(cfg_path).stem}")
    if seed == cfg["seed"]:
        return cfg_path, Path(base_dir)
    c = copy.deepcopy(cfg)
    c["seed"] = seed
    c["training"]["checkpoint_dir"] = f"{base_dir}_seed{seed}"
    p = Path(cfg_path).parent / "seeds" / f"{Path(cfg_path).stem}_seed{seed}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
    return str(p), Path(c["training"]["checkpoint_dir"])


def train(cfg_path, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = ckpt_dir / "train.log"
    print(f"    training -> {log}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, "src/training/train.py", cfg_path],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in p.stdout:
            f.write(line)
            progress_bar = "it/s]" in line or "s/it]" in line or "%|" in line
            if not progress_bar and (line.startswith(("Epoch", "  →", "Reached", "Traceback",
                                                      "Parameters", "Training loop"))
                                     or "Error" in line):
                print("     ", line.rstrip())
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"training failed (exit {p.returncode}); see {log}")
    return time.time() - t0


def evaluate(cfg_path, cfg, ckpt, test):
    """BLEU/chrF of one checkpoint, cached next to it (recomputed if best.pt changes)."""
    cache = ckpt.parent / f"{args.split}.eval.json"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("ckpt_mtime") == ckpt.stat().st_mtime:
            return c
    model, tok, cfg_m, meta, _, device = load_model(cfg_path, ckpt)
    m = meta.get("config", cfg_m)["model"]
    restrict = cfg.get("inference", {}).get("restrict_target_vocab", False)
    kw = dict(max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"])
    res = evaluate_bleu(model, tok, test["fr"], test["en"], device, restrict_output=restrict, **kw)
    out = {"bleu": res["bleu"], "chrf": res["chrf"], "step": meta.get("step"),
           "restricted": restrict, "signature": res["bleu_signature"],
           "train_seconds_to_best": meta.get("train_seconds"), "ckpt_mtime": ckpt.stat().st_mtime}
    (ckpt.parent / f"{args.split}.hyp.en").write_text("\n".join(res["hypotheses"]) + "\n", encoding="utf-8")
    if restrict:  # also score without the output mask, for an apples-to-apples view
        res2 = evaluate_bleu(model, tok, test["fr"], test["en"], device, restrict_output=False, **kw)
        out.update(bleu_unrestricted=res2["bleu"], chrf_unrestricted=res2["chrf"])
    del model
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def fmt(m, sd, prec=2):
    if m is None:
        return "-"
    return f"{m:.{prec}f}" + (f"±{sd:.{prec}f}" if sd is not None else "")


rows, seen = [], {}
for cfg_path in configs:
    cfg = load_config(cfg_path)
    name = Path(cfg_path).stem
    label = cfg.get("benchmark_label", name)
    print(f"\n=== {label}  ({cfg_path})")

    tok = load_tokenizer(cfg)
    data_dir = Path(cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}"))
    test = {lang: read_lines(data_dir / f"{args.split}.{lang}") for lang in ("fr", "en")}
    row = {"name": name, "label": label, "vocab": tok.vocab_size}
    for lang, ls in test.items():
        ids = [tok.encode(l, lang=lang) for l in ls]
        row[f"{lang}_tok"] = sum(map(len, ids)) / len(ls)
        row[f"{lang}_unk"] = sum(x.count(1) for x in ids)  # <unk> is id 1 for every tokenizer here

    sig = signature(tok, {k: v[:300] for k, v in test.items()})
    if sig in seen:
        prev = seen[sig]
        print(f"  encodes exactly like {prev['label']}: same model, reusing its result")
        rows.append({**prev, "name": name, "label": label, "duplicate_of": prev["label"]})
        continue

    row.update(utilization(tok, data_dir))
    print(f"  vocab used on train: {row['util_used']:.1%} (>= {WELL_TRAINED}x: {row['util_well']:.1%}, "
          f"both languages: {row['util_shared']:.1%}, never used: {row['unused_tokens']})")

    runs = []
    for seed in args.seeds or [cfg["seed"]]:
        run_cfg, ckpt_dir = seed_config(cfg_path, cfg, seed)
        ckpt = ckpt_dir / "best.pt"
        print(f"  seed {seed}: {ckpt_dir}")
        if not ckpt.exists():
            if args.no_train:
                print("    no checkpoint, skipping (--no-train)")
                continue
            train(run_cfg, ckpt_dir)
        try:
            r = evaluate(run_cfg, cfg, ckpt, test)
        except ValueError as e:  # checkpoint trained with a different vocab
            print(f"    {e}\n    -> delete {ckpt_dir} to retrain this run")
            continue
        summary = ckpt_dir / "train_summary.json"
        if summary.exists():
            s = json.loads(summary.read_text(encoding="utf-8"))
            r.update({k: s.get(k) for k in ("steps_per_second", "tgt_tokens_per_second", "gpu",
                                            "train_seconds", "train_seconds_to_best")})
        r["seed"] = seed
        runs.append(r)
        print(f"    BLEU {r['bleu']:.2f} | chrF {r['chrf']:.2f} (best step {r['step']})")
    if not runs:
        continue

    row["runs"] = runs
    row["restricted"] = runs[0]["restricted"]
    for k in ("bleu", "chrf", "bleu_unrestricted", "chrf_unrestricted", "step",
              "steps_per_second", "tgt_tokens_per_second"):
        row[k], row[k + "_sd"] = mean_sd([r.get(k) for r in runs])
    m, sd = mean_sd([r["train_seconds_to_best"] / 60 if r.get("train_seconds_to_best") else None
                     for r in runs])
    row["min_to_best"], row["min_to_best_sd"] = m, sd
    rows.append(row)
    seen[sig] = row

# ---------------------------------------------------------------- report
rows = [r for r in rows if "bleu" in r]
n_seeds = max((len(r["runs"]) for r in rows), default=0)
print(f"\n\nResults on {args.split} ({len(test['fr'])} sentences), same model and training for every row"
      + (f", mean±std over up to {n_seeds} seeds" if n_seeds > 1 else ""))


def label_of(r):
    return (r["label"] + (" *" if r.get("restricted") else "")
            + (f" (= {r['duplicate_of']})" if r.get("duplicate_of") else ""))


# 1. Quality
hdr = f"{'Tokenizer':<30}{'seeds':>6}{'BLEU':>14}{'chrF':>14}{'best step':>12}"
print("\n" + hdr + "\n" + "-" * len(hdr))
md = ["## Translation quality", "",
      "| Tokenizer | seeds | BLEU | chrF | best step | BLEU per seed |", "|---|---|---|---|---|---|"]
for r in rows:
    per_seed = ", ".join(f"{x['bleu']:.2f}" for x in r["runs"])
    print(f"{label_of(r)[:30]:<30}{len(r['runs']):>6}{fmt(r['bleu'], r['bleu_sd']):>14}"
          f"{fmt(r['chrf'], r['chrf_sd']):>14}{fmt(r['step'], None, 0):>12}   [{per_seed}]")
    md.append(f"| {label_of(r)} | {len(r['runs'])} | {fmt(r['bleu'], r['bleu_sd'])} | "
              f"{fmt(r['chrf'], r['chrf_sd'])} | {fmt(r['step'], None, 0)} | {per_seed} |")
    if r.get("bleu_unrestricted") is not None:
        print(f"{'  same, no output mask':<30}{'':>6}{fmt(r['bleu_unrestricted'], r['bleu_unrestricted_sd']):>14}"
              f"{fmt(r['chrf_unrestricted'], r['chrf_unrestricted_sd']):>14}")
        md.append(f"| {r['label']} (no output mask) | | {fmt(r['bleu_unrestricted'], r['bleu_unrestricted_sd'])} | "
                  f"{fmt(r['chrf_unrestricted'], r['chrf_unrestricted_sd'])} | | |")

# 2. Tokenization and vocabulary utilization
hdr = (f"{'Tokenizer':<30}{'vocab':>7}{'FR tok/s':>10}{'EN tok/s':>10}{'<unk>':>7}"
       f"{'used':>8}{f'>={WELL_TRAINED}x':>8}{'FR+EN':>8}{'unused':>8}")
print("\n" + hdr + "\n" + "-" * len(hdr))
md += ["", "## Tokenization and vocabulary utilization (training set)", "",
       f"| Tokenizer | vocab | FR tokens/sent | EN tokens/sent | <unk> | used | used >={WELL_TRAINED}x | "
       "used in FR and EN | never used |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    unk = r["fr_unk"] + r["en_unk"]
    print(f"{label_of(r)[:30]:<30}{r['vocab']:>7,}{r['fr_tok']:>10.1f}{r['en_tok']:>10.1f}{unk:>7}"
          f"{r['util_used']:>8.1%}{r['util_well']:>8.1%}{r['util_shared']:>8.1%}{r['unused_tokens']:>8,}")
    md.append(f"| {label_of(r)} | {r['vocab']:,} | {r['fr_tok']:.1f} | {r['en_tok']:.1f} | {unk} | "
              f"{r['util_used']:.1%} | {r['util_well']:.1%} | {r['util_shared']:.1%} | {r['unused_tokens']:,} |")

# 3. Training speed
gpus = {x.get("gpu") for r in rows for x in r["runs"] if x.get("gpu")}
hdr = f"{'Tokenizer':<30}{'steps/s':>12}{'tgt tok/s':>14}{'min to best':>14}"
print("\n" + hdr + "\n" + "-" * len(hdr))
md += ["", "## Training speed", "", "| Tokenizer | steps/s | target tokens/s | minutes to best checkpoint |",
       "|---|---|---|---|"]
for r in rows:
    print(f"{label_of(r)[:30]:<30}{fmt(r['steps_per_second'], r['steps_per_second_sd']):>12}"
          f"{fmt(r['tgt_tokens_per_second'], r['tgt_tokens_per_second_sd'], 0):>14}"
          f"{fmt(r['min_to_best'], r['min_to_best_sd'], 1):>14}")
    md.append(f"| {label_of(r)} | {fmt(r['steps_per_second'], r['steps_per_second_sd'])} | "
              f"{fmt(r['tgt_tokens_per_second'], r['tgt_tokens_per_second_sd'], 0)} | "
              f"{fmt(r['min_to_best'], r['min_to_best_sd'], 1)} |")
if not gpus:
    print("(no timing yet: runs trained before train_summary.json existed show '-')")
elif len(gpus) > 1:
    print(f"WARNING: runs used different GPUs {sorted(gpus)}; speeds are not comparable")
else:
    print(f"GPU: {gpus.pop()}")

print("\n* decoder restricted to tokens seen in English (only possible with per-language counts)")
if n_seeds > 1:
    print("A gap smaller than about 2 std of the runs is not a real difference.")
else:
    print("One training run per tokenizer: BLEU differences under ~0.5-1 point can be seed noise.")

out = Path("logs")
out.mkdir(exist_ok=True)
(out / f"tokenizer_benchmark_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(out / f"tokenizer_benchmark_{args.split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nSaved logs/tokenizer_benchmark_{args.split}.md and .json")
