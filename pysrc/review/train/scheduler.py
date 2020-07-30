from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR


class CustomScheduler(_LRScheduler):
    timestep: int = 0

    def __init__(self, optimizer, gamma, warmup=None):
        self.optimizer = optimizer
        self.after_warmup = ExponentialLR(optimizer, gamma=gamma)
        self.initial_lrs = [p_group['lr'] for p_group in self.optimizer.param_groups]
        self.warmup = 0 if warmup is None else warmup
        super(CustomScheduler, self).__init__(optimizer)

    def get_lr(self):
        return [self.timestep * group_init_lr / self.warmup for group_init_lr in self.initial_lrs] \
            if self.timestep < self.warmup else self.after_warmup.get_lr()

    def step(self, epoch=None):
        if self.timestep < self.warmup:
            self.timestep += 1
            super(CustomScheduler, self).step(epoch)
        else:
            self.after_warmup.step(epoch)


class NoamScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup):
        assert warmup > 0
        self.optimizer = optimizer
        self.initial_lrs = [p_group['lr'] for p_group in self.optimizer.param_groups]
        self.warmup = warmup
        self.timestep = 0
        super(NoamScheduler, self).__init__(optimizer)

    def get_lr(self):
        noam_lr = self.get_noam_lr()
        return [group_init_lr * noam_lr for group_init_lr in self.initial_lrs]

    def get_noam_lr(self):
        return min(self.timestep ** -0.5, self.timestep * self.warmup ** -1.5)

    def step(self, epoch=None):
        self.timestep += 1
        super(NoamScheduler, self).step(epoch)


if __name__ == "__main__":
    """
    some test't
    """

    import torch.nn as nn
    from pytorch_transformers.optimization import AdamW

    wrmp = 5
    net = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 10),
    )
    opt = AdamW([
        {'params': [p for n, p in net.named_parameters() if n.startswith('0')], 'lr': 0.05},
        {'params': [p for n, p in net.named_parameters() if n.startswith('1')], 'lr': 0.0001}
    ])

    scheduler = NoamScheduler(opt, warmup=wrmp)

    for i in range(15):
        print(f"epoch: {i + 1}")
        print(*[param['lr'] for param in opt.param_groups], sep=' | ', end='\n\n')
        scheduler.step()
