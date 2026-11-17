"""Train the same transformer with each tokenizer and compare translation quality.

    python scripts/train_baseline_tokenizers.py      # once: builds tokenizers + configs
    python scripts/benchmark_tokenizers.py           # trains what's missing, evaluates all

Every config in --configs differs ONLY in its tokenizer (same model, data, seed,
steps). Each run is trained in its own process; a run whose checkpoint already
exists is not retrained, so the script can be re-run after a Colab disconnect
and continues where it stopped. Tokenizers that encode identically (the retrained
M2M-100 and NLLB do) are detected and evaluated once.

Results: printed table + logs/tokenizer_benchmark_<split>.md / .json
"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import evaluate_bleu
from src.inference.translator import load_model
from src.tokenizer.loader import load_tokenizer
from src.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="configs/tokenizers/thin_slice_*.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--no-train", action="store_true", help="only evaluate runs that already have a checkpoint")
ap.add_argument("--only", nargs="*", help="config names (e.g. thin_slice_gpt2) to include")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

configs = sorted(glob.glob(args.configs), key=lambda p: (not p.endswith("_ours.yaml"), p))
if args.only:
    configs = [c for c in configs if Path(c).stem in args.only]
if not configs:
    sys.exit(f"No configs match {args.configs}. Run scripts/train_baseline_tokenizers.py first.")


def signature(tok, lines):
    h = hashlib.sha1()
    for lang, ls in lines.items():
        for l in ls:
            h.update(str(tok.encode(l, lang=lang)).encode())
    return h.hexdigest()


def train(cfg_path, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = ckpt_dir / "train.log"
    print(f"  training -> {log}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, "src/training/train.py", cfg_path],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line)
            progress_bar = "it/s]" in line or "s/it]" in line or "%|" in line
            if not progress_bar and (line.startswith(("Epoch", "  →", "Reached", "Traceback", "Parameters"))
                                     or "Error" in line):
                print("   ", line.rstrip())
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"training failed (exit {p.returncode}); see {log}")
    return time.time() - t0


rows, seen = [], {}
for cfg_path in configs:
    cfg = load_config(cfg_path)
    name = Path(cfg_path).stem
    label = cfg.get("benchmark_label", name)
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", f"checkpoints/{name}"))
    ckpt = ckpt_dir / "best.pt"
    print(f"\n=== {label}  ({cfg_path})")

    tok = load_tokenizer(cfg)
    data_dir = Path(cfg["data"].get("data_dir", f"data/raw/{cfg['data']['name']}"))
    test = {lang: [l.strip() for l in open(data_dir / f"{args.split}.{lang}", encoding="utf-8")]
            for lang in ("fr", "en")}
    row = {"name": name, "label": label, "vocab": tok.vocab_size}
    for lang, ls in test.items():
        ids = [tok.encode(l, lang=lang) for l in ls]
        row[f"{lang}_tok"] = sum(map(len, ids)) / len(ls)
        row[f"{lang}_unk"] = sum(x.count(1) for x in ids)  # <unk> is id 1 for every tokenizer here

    sig = signature(tok, {k: v[:300] for k, v in test.items()})
    if sig in seen:
        prev = seen[sig]
        print(f"  encodes exactly like {prev['label']}: same model, reusing its result")
        rows.append({**prev, **{k: row[k] for k in ("name", "label")}, "duplicate_of": prev["label"]})
        continue

    if not ckpt.exists():
        if args.no_train:
            print(f"  no checkpoint at {ckpt}, skipping (--no-train)")
            continue
        row["train_minutes"] = train(cfg_path, ckpt_dir) / 60

    try:
        model, tok, cfg_m, meta, _, device = load_model(cfg_path, ckpt)
    except ValueError as e:  # checkpoint trained with a different vocab
        print(f"  {e}\n  -> delete {ckpt_dir} to retrain this run")
        continue
    m = meta.get("config", cfg_m)["model"]
    restrict = cfg.get("inference", {}).get("restrict_target_vocab", False)
    kw = dict(max_len=m["max_seq_len"] - 2, max_src_len=m["max_seq_len"])
    res = evaluate_bleu(model, tok, test["fr"], test["en"], device, restrict_output=restrict, **kw)
    row.update(bleu=res["bleu"], chrf=res["chrf"], step=meta.get("step"), restricted=restrict,
               signature=res["bleu_signature"])
    (ckpt_dir / f"{args.split}.hyp.en").write_text("\n".join(res["hypotheses"]) + "\n", encoding="utf-8")
    if restrict:  # also report ours without the output mask, for an apples-to-apples view
        res2 = evaluate_bleu(model, tok, test["fr"], test["en"], device, restrict_output=False, **kw)
        row.update(bleu_unrestricted=res2["bleu"], chrf_unrestricted=res2["chrf"])
    print(f"  BLEU {row['bleu']:.2f} | chrF {row['chrf']:.2f} (best step {row['step']})")
    rows.append(row)
    seen[sig] = row

# ---------------------------------------------------------------- report
print(f"\n\nResults on {args.split} ({len(test['fr'])} sentences), same model and training for every row")
hdr = f"{'Tokenizer':<26}{'vocab':>7}{'FR tok/s':>10}{'EN tok/s':>10}{'<unk>':>7}{'BLEU':>8}{'chrF':>8}{'step':>7}"
print(hdr + "\n" + "-" * len(hdr))
md = ["| Tokenizer | Vocab | FR tokens/sent | EN tokens/sent | <unk> | BLEU | chrF | best step |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "bleu" not in r:
        continue
    lab = r["label"] + (" *" if r.get("restricted") else "") + (" (= " + r["duplicate_of"] + ")" if r.get("duplicate_of") else "")
    unk = r["fr_unk"] + r["en_unk"]
    print(f"{lab[:26]:<26}{r['vocab']:>7,}{r['fr_tok']:>10.1f}{r['en_tok']:>10.1f}{unk:>7}"
          f"{r['bleu']:>8.2f}{r['chrf']:>8.2f}{r['step'] or 0:>7}")
    md.append(f"| {lab} | {r['vocab']:,} | {r['fr_tok']:.1f} | {r['en_tok']:.1f} | {unk} | "
              f"{r['bleu']:.2f} | {r['chrf']:.2f} | {r['step']} |")
    if "bleu_unrestricted" in r:
        print(f"{'  same, no output mask':<26}{'':>7}{'':>10}{'':>10}{'':>7}"
              f"{r['bleu_unrestricted']:>8.2f}{r['chrf_unrestricted']:>8.2f}")
        md.append(f"| {r['label']} (no output mask) | | | | | {r['bleu_unrestricted']:.2f} | "
                  f"{r['chrf_unrestricted']:.2f} | |")
print("* decoder restricted to tokens seen in English (only possible with per-language counts)")
print("One training run per tokenizer: BLEU differences under ~0.5-1 point can be seed noise.")

out = Path("logs")
out.mkdir(exist_ok=True)
(out / f"tokenizer_benchmark_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(out / f"tokenizer_benchmark_{args.split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nSaved logs/tokenizer_benchmark_{args.split}.md and .json")
