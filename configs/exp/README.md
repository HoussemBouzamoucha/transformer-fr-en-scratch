# Experiments

`00_paper_base.yaml` is the "Attention Is All You Need" base model (FR -> EN);
its header lists what is faithful to the paper and what isn't. Every later
experiment is measured against it.

## Adding a change (RoPE, flash attention, Pre-LN, ...)

1. Implement it **next to** the original, behind a switch in `ARCH_DEFAULTS` /
   `ARCH_CHOICES` (`src/model/transformer.py`). Never replace the old code path,
   and keep the default equal to the old behaviour, so every saved checkpoint
   still rebuilds from the config stored inside it.
2. Copy the previous config to `NN_<change>.yaml` and change **one** setting.
3. Train with the same data, tokenizer, step budget and seeds:
   `python src/training/train.py configs/exp/NN_<change>.yaml`
   (checkpoints go to `checkpoints/NN_<change>/`, so runs never overwrite each other).
4. Score the averaged model with beam search:
   `python evaluate.py --config configs/exp/NN_<change>.yaml --checkpoint checkpoints/NN_<change>/averaged.pt`
5. Tag the commit that produced the result: `git tag exp-NN-<change>`.

Seed noise on Multi30k is about 0.5-1 BLEU, so run 2-3 seeds before calling a
difference real. For speed-only changes (flash attention), compare tokens/s from
`train_summary.json` and check the outputs match the old attention numerically.
