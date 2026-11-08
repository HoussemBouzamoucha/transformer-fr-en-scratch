import torch
import torch.nn as nn


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        """
        logits: (batch * seq_len, vocab_size)
        target: (batch * seq_len,)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth_target[:, self.pad_id] = 0.0

        pad_mask = (target != self.pad_id).unsqueeze(1).float()
        loss = -(smooth_target * log_probs * pad_mask).sum(dim=-1)

        n_tokens = pad_mask.sum()
        return loss.sum() / n_tokens