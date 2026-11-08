# warmup + inverse-sqrt LRimport torch.optim as optim


class InverseSqrtScheduler:
    """
    Warmup then inverse-sqrt decay, as in the original Transformer paper.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        s = self.step_num
        w = self.warmup_steps
        return (self.d_model ** -0.5) * min(s ** -0.5, s * w ** -1.5)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]