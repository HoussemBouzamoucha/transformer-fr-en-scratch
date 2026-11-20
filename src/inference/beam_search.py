"""Batched beam search with the GNMT length penalty.

Vaswani et al. 2017 sec. 6.1: "We used beam search with a beam size of 4 and
length penalty alpha = 0.6. ... We set the maximum output length during
inference to input length + 50, but terminate early when possible."

The length penalty is Wu et al. 2016's lp(Y) = ((5 + |Y|) / 6) ** alpha; a
hypothesis is ranked by log P(Y|X) / lp(Y). Stopping follows tensor2tensor (the
paper's code): a sentence is done once it has beam_size finished hypotheses and
the worst of them beats the best score any live beam could still reach.
"""
import torch


def length_penalty(length, alpha):
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(model, src, bos_id, eos_id, beam_size=4, alpha=0.6, extra_len=50,
                max_len=None, allowed_ids=None, pad_id=None):
    """src: (batch, src_len) ids. Returns a list of id lists (no <s>, no </s>).

    extra_len: each sentence may produce up to (its source length + extra_len) tokens.
    max_len:   hard cap on output tokens (the positional table size).
    allowed_ids: optional ids the decoder may emit (e.g. English tokens only).
    """
    model.eval()
    device = src.device
    B, K = src.size(0), beam_size
    pad_id = model.pad_id if pad_id is None else pad_id

    # Per-sentence output limit: input length (real tokens, without <s>/</s>) + extra_len.
    src_len = (src != pad_id).sum(1) - 2
    limits = (src_len + extra_len).clamp(min=1)
    if max_len is not None:
        limits = limits.clamp(max=max_len)
    limits = limits.tolist()
    T = max(limits)

    memory, src_mask = model.encode(src)
    memory = memory.repeat_interleave(K, dim=0)        # (B*K, S, d)
    src_mask = src_mask.repeat_interleave(K, dim=0)

    vocab = model.output_projection.out_features
    mask = None
    if allowed_ids is not None:
        mask = torch.full((vocab,), float("-inf"), device=device)
        mask[torch.as_tensor(allowed_ids, device=device)] = 0.0
        mask[eos_id] = 0.0

    seqs = torch.full((B * K, 1), bos_id, dtype=torch.long, device=device)
    # Only beam 0 is live at the start, so the first step doesn't pick K copies of one token.
    scores = torch.full((B, K), float("-inf"), device=device)
    scores[:, 0] = 0.0

    finished = [[] for _ in range(B)]   # per sentence: (normalized score, ids)
    done = [False] * B

    for t in range(1, T + 1):
        out = model.decode(seqs, memory, src_mask)
        logp = torch.log_softmax(model.output_projection(out[:, -1, :]).float(), dim=-1)
        if mask is not None:
            logp = logp + mask
        cand = (scores.unsqueeze(-1) + logp.view(B, K, vocab)).view(B, K * vocab)
        top_scores, top_idx = cand.topk(2 * K, dim=-1)   # at most K of these end in </s>
        top_beam, top_tok = top_idx // vocab, top_idx % vocab
        lp_t = length_penalty(t, alpha)
        ts, tb, tt = top_scores.tolist(), top_beam.tolist(), top_tok.tolist()
        prev = seqs.view(B, K, -1)
        nb, nt, ns = [], [], []   # chosen (beam, token, score) per sentence, K each

        for b in range(B):
            beams, toks, scs = [], [], []
            if not done[b]:
                for s, beam, tok in zip(ts[b], tb[b], tt[b]):
                    if s == float("-inf"):
                        break
                    if tok == eos_id:
                        finished[b].append((s / lp_t, prev[b, beam, 1:].tolist()))
                    elif len(beams) < K:
                        beams.append(beam); toks.append(tok); scs.append(s)
                finished[b] = sorted(finished[b], key=lambda h: -h[0])[:K]

                if t >= limits[b]:
                    # Out of length: live beams count as finished at their current length.
                    for beam, tok, s in zip(beams, toks, scs):
                        finished[b].append((s / lp_t, prev[b, beam, 1:].tolist() + [tok]))
                    done[b] = True
                elif not beams:
                    done[b] = True
                elif len(finished[b]) == K:
                    # Live scores only decrease and lp is largest at the limit, so
                    # this is the best normalized score a live beam can still reach.
                    done[b] = finished[b][-1][0] >= scs[0] / length_penalty(limits[b], alpha)
            if done[b]:
                # Finished sentences keep decoding (batched) but are ignored.
                beams, toks, scs = [0], [eos_id], [0.0]
            pad = K - len(beams)
            nb.append(beams + [0] * pad)
            nt.append(toks + [eos_id] * pad)
            ns.append(scs + [float("-inf")] * pad)

        if all(done):
            break
        new_beam = torch.tensor(nb, device=device)
        flat = (torch.arange(B, device=device).unsqueeze(1) * K + new_beam).view(-1)
        seqs = torch.cat([seqs[flat], torch.tensor(nt, device=device).view(-1, 1)], dim=1)
        scores = torch.tensor(ns, device=device)

    return [max(f, key=lambda h: h[0])[1] if f else [] for f in finished]
