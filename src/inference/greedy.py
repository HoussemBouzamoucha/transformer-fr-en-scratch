import torch


@torch.no_grad()
def greedy_decode(model, src, bos_id, eos_id, max_len=100, allowed_ids=None):
    """Batched greedy decoding. Returns (batch, <=max_len) ids without the leading <s>.

    allowed_ids: optional list of token ids the decoder may emit (e.g. English
    tokens only, from tokenizer.target_token_ids()); everything else is masked.
    """
    model.eval()
    device = src.device
    memory, src_mask = model.encode(src)
    tgt = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=device)

    mask = None
    if allowed_ids is not None:
        vocab = model.output_projection.out_features
        mask = torch.full((vocab,), float("-inf"), device=device)
        mask[torch.as_tensor(allowed_ids, device=device)] = 0.0

    for _ in range(max_len):
        out = model.decode(tgt, memory, src_mask)
        logits = model.output_projection(out[:, -1, :]).float()
        if mask is not None:
            logits = logits + mask
        next_tokens = logits.argmax(-1).masked_fill(done, eos_id)
        tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)
        done |= next_tokens == eos_id
        if done.all():
            break
    return tgt[:, 1:]


def strip_after_eos(ids, eos_id):
    out = []
    for t in ids:
        if t == eos_id:
            break
        out.append(t)
    return out
