import torch
import sacrebleu
from tqdm import tqdm


def greedy_decode(model, src, bos_id, eos_id, device, max_len=100):
    model.eval()
    with torch.no_grad():
        memory, src_mask = model.encode(src)
        tgt = torch.tensor([[bos_id]] * src.size(0)).to(device)

        done = torch.zeros(src.size(0), dtype=torch.bool).to(device)
        for _ in range(max_len):
            out = model.decode(tgt, memory, src_mask)
            logits = model.output_projection(out[:, -1, :])
            next_tokens = logits.argmax(-1)
            next_tokens = next_tokens.masked_fill(done, eos_id)
            tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)
            done = done | (next_tokens == eos_id)
            if done.all():
                break

    return tgt[:, 1:]


def decode_tokens(token_ids, sp, eos_id):
    result = []
    for ids in token_ids.tolist():
        trimmed = []
        for t in ids:
            if t == eos_id:
                break
            trimmed.append(t)
        result.append(sp.decode(trimmed))
    return result


def evaluate_bleu(model, data_loader, sp, bos_id, eos_id, device, max_batches=None):
    hypotheses = []
    references = []

    for i, (src, tgt) in enumerate(tqdm(data_loader, desc="Evaluating")):
        if max_batches and i >= max_batches:
            break

        src = src.to(device)
        preds = greedy_decode(model, src, bos_id, eos_id, device)
        hyps = decode_tokens(preds, sp, eos_id)
        refs = decode_tokens(tgt, sp, eos_id)

        hypotheses.extend(hyps)
        references.extend(refs)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "chrf": chrf.score,
        "hypotheses": hypotheses,
        "references": references,
    }