"""BLEU / chrF for FR -> EN.

Sources are encoded exactly like training (lang="fr", same threshold); the
model's ids are turned back into text with the tokenizer; and scores are
computed against the raw reference sentences, never against references
rebuilt from token ids (that would hide <unk>s and normalization changes).
"""
import torch
import sacrebleu

from src.inference.beam_search import beam_search
from src.inference.greedy import greedy_decode, strip_after_eos


def translate(model, tokenizer, src_lines, device, batch_size=64, max_len=100,
              restrict_output=True, max_src_len=None, beam_size=1, length_penalty=0.6,
              extra_len=50):
    """Translate raw French lines; returns English strings in the same order.

    beam_size=1 decodes greedily (up to max_len tokens). beam_size>1 runs beam
    search with the GNMT length penalty, up to source length + extra_len tokens
    (capped at max_len), as in the paper (sec. 6.1).
    """
    allowed = tokenizer.target_token_ids("en") if restrict_output else None
    pad, bos, eos = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id

    encoded = []
    for line in src_lines:
        ids = [bos] + tokenizer.encode(line, lang="fr") + [eos]
        if max_src_len and len(ids) > max_src_len:  # positional table limit
            ids = ids[: max_src_len - 1] + [eos]
        encoded.append(ids)

    # Sort by length so batches have little padding, then restore order.
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    hyps = [None] * len(encoded)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [torch.tensor(encoded[i]) for i in idx]
        src = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad).to(device)
        if beam_size > 1:
            out = beam_search(model, src, bos, eos, beam_size=beam_size, alpha=length_penalty,
                              extra_len=extra_len, max_len=max_len, allowed_ids=allowed, pad_id=pad)
        else:
            out = greedy_decode(model, src, bos, eos, max_len=max_len, allowed_ids=allowed).tolist()
        for i, ids in zip(idx, out):
            hyps[i] = tokenizer.decode(strip_after_eos(ids, eos))
    return hyps


def evaluate_bleu(model, tokenizer, src_lines, ref_lines, device, **kwargs):
    hyps = translate(model, tokenizer, src_lines, device, **kwargs)
    bleu = sacrebleu.corpus_bleu(hyps, [ref_lines])
    chrf = sacrebleu.corpus_chrf(hyps, [ref_lines])
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "chrf": chrf.score,
        "hypotheses": hyps,
        "references": ref_lines,
    }


def decoding_options(cfg: dict) -> dict:
    """translate() keyword arguments from a config's `inference:` block.

    Configs without beam settings decode greedily, as before beam search existed.
    """
    inf = cfg.get("inference", {})
    return {
        "beam_size": inf.get("beam_size", 1),
        "length_penalty": inf.get("length_penalty", 0.6),
        "extra_len": inf.get("max_len_extra", 50),
    }
