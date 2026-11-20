"""Retrain the GPT-2 / XLM-R / M2M-100 / NLLB tokenizer algorithms on OUR data.

    python scripts/train_baseline_tokenizers.py                      # Multi30k, 8k vocab
    python scripts/train_baseline_tokenizers.py --data-dir data/clean/wmt14 --base-config configs/wmt14.yaml

For a fair comparison every tokenizer gets:
  - exactly the same training sentences as our joint BPE (same sampler, seed, size)
  - the same vocab size
  - the same special tokens with the same ids: <pad>=0 <unk>=1 <s>=2 </s>=3
  - its ORIGINAL algorithm and settings, read from the released tokenizer:
      GPT-2   : byte-level BPE (its pre-tokenizer regex + byte decoder, no normalizer)
      XLM-R   : SentencePiece Unigram, nmt_nfkc
      M2M-100 : SentencePiece BPE, nmt_nfkc
      NLLB    : SentencePiece BPE, nmt_nfkc (same settings as M2M-100)
Only their training data and vocab size change. Writes one model config per tokenizer
to configs/tokenizers/ (copies of --base-config with only the tokenizer swapped).

Needs: pip install transformers tokenizers sentencepiece "protobuf>=5"
"""
import argparse
import copy
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.tokenizer.sampling import sample_pairs
from src.utils import load_config

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
SPM_MODELS = {"xlmr": "xlm-roberta-base", "m2m100": "facebook/m2m100_418M",
              "nllb": "facebook/nllb-200-distilled-600M"}
LABELS = {"ours": "Our joint BPE", "gpt2": "GPT-2 (byte-level BPE)", "xlmr": "XLM-R (SPM unigram)",
          "m2m100": "M2M-100 (SPM BPE)", "nllb": "NLLB (SPM BPE)"}

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/raw/multi30k")
ap.add_argument("--base-config", default="configs/thin_slice.yaml")
ap.add_argument("--vocab-size", type=int, default=None, help="default: tokenizer.vocab_size of the base config")
ap.add_argument("--sample-size", type=int, default=100_000, help="same as train_own_bpe.py")
ap.add_argument("--seed", type=int, default=42, help="same as train_own_bpe.py")
ap.add_argument("--out-dir", default=None, help="default: data/vocab/baselines/<data name>")
ap.add_argument("--config-dir", default="configs/tokenizers")
ap.add_argument("--only", nargs="*", choices=["gpt2", *SPM_MODELS])
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

base_cfg = load_config(args.base_config)
vocab_size = args.vocab_size or base_cfg["tokenizer"]["vocab_size"]
data_name = Path(args.data_dir).name
out_dir = Path(args.out_dir or f"data/vocab/baselines/{data_name}")
out_dir.mkdir(parents=True, exist_ok=True)
cfg_dir = Path(args.config_dir)
cfg_dir.mkdir(parents=True, exist_ok=True)
todo = args.only or ["gpt2", *SPM_MODELS]

corpora, n = sample_pairs(args.data_dir, args.sample_size, args.seed)
lines = corpora["fr"] + corpora["en"]
print(f"Training on {len(corpora['fr']):,} pairs ({len(lines):,} lines) from {args.data_dir}, "
      f"vocab {vocab_size}")

made = {}

if "gpt2" in todo:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import AutoTokenizer
    orig = AutoTokenizer.from_pretrained("gpt2").backend_tokenizer
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = orig.normalizer          # None for GPT-2
    tok.pre_tokenizer = orig.pre_tokenizer    # ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = orig.decoder                # ByteLevel
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(lines, trainer=trainer)
    assert [tok.token_to_id(t) for t in SPECIALS] == [0, 1, 2, 3]
    path = out_dir / f"gpt2_bytebpe_{vocab_size // 1000}k.json"
    tok.save(str(path))
    made["gpt2"] = {"type": "hf", "path": path.as_posix()}
    print(f"  GPT-2 byte-level BPE -> {path} ({tok.get_vocab_size()} tokens)")

spm_todo = [k for k in SPM_MODELS if k in todo]
if spm_todo:
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for key in spm_todo:
            repo = SPM_MODELS[key]
            slow = AutoTokenizer.from_pretrained(repo, use_fast=False)
            spm_file = getattr(slow, "spm_file", None) or slow.vocab_file
            proto = pb.ModelProto()
            proto.ParseFromString(Path(spm_file).read_bytes())
            ts, ns = proto.trainer_spec, proto.normalizer_spec
            model_type = pb.TrainerSpec.ModelType.Name(ts.model_type).lower()
            prefix = out_dir / f"{key}_spm{model_type}_{vocab_size // 1000}k"
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
                # algorithm + settings copied from the released model
                model_type=model_type,
                character_coverage=ts.character_coverage,
                split_by_number=ts.split_by_number,
                split_digits=ts.split_digits,
                byte_fallback=ts.byte_fallback,
                max_sentencepiece_length=ts.max_sentencepiece_length,
                normalization_rule_name=ns.name,
                add_dummy_prefix=ns.add_dummy_prefix,
                remove_extra_whitespaces=ns.remove_extra_whitespaces,
                # same special ids as every other tokenizer here
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
                minloglevel=2,
            )
            path = prefix.with_suffix(".model")
            made[key] = {"type": "sentencepiece", "path": path.as_posix()}
            print(f"  {repo}: {model_type}, coverage {ts.character_coverage:.6f}, "
                  f"normalizer {ns.name} -> {path}")

# One config per tokenizer: identical model/training settings, only the tokenizer differs.
stem = Path(args.base_config).stem
ours = copy.deepcopy(base_cfg)
ours.setdefault("training", {})
# Our tokenizer with the base config IS the base run: reuse its checkpoint if present.
ours["training"].setdefault("checkpoint_dir", f"checkpoints/{stem}")
written = {"ours": ours}
for key, tok_cfg in made.items():
    cfg = copy.deepcopy(base_cfg)
    cfg["tokenizer"] = {"type": tok_cfg["type"], "path": tok_cfg["path"], "vocab_size": vocab_size}
    cfg["inference"] = {"restrict_target_vocab": False}  # no per-language counts
    cfg.setdefault("training", {})["checkpoint_dir"] = f"checkpoints/{stem}_tok_{key}"
    written[key] = cfg
for key, cfg in written.items():
    cfg["benchmark_label"] = LABELS[key]
    p = cfg_dir / f"{stem}_{key}.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"  config -> {p}")
