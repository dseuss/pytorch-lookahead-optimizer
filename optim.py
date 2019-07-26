import itertools as it
import math
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer


class LookaheadOptimizer(Optimizer):
    def __init__(self, params, fast_optimizer=torch.optim.SGD,
                 slow_update_rate=0.5, lookahead_steps=6, **kwargs):
        if not 0.0 <= slow_update_rate <= 1.0:
            raise ValueError(f'Invalid slow_update_rate: {slow_update_rate}')
        if not 1 <= lookahead_steps:
            raise ValueError(f'Invalid lookahead_steps: {lookahead_steps}')
        if not issubclass(fast_optimizer, Optimizer):
            raise ValueError(f'Invalid fast_optimizer: {fast_optimizer}')

        params = list(params)
        self.fast_optimizer = fast_optimizer(params, **kwargs)

        # TODO Proper handling of params for fast optimizer
        defaults = dict(
            **self.fast_optimizer.defaults,
            slow_update_rate=slow_update_rate,
            lookahead_steps=lookahead_steps,
            step_counter=0)

        super().__init__(params=params, defaults=defaults)
        self.slow_weights = [[w.clone().detach() for w in g['params']]
                             for g in self.param_groups]
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.fast_optimizer.step()

        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] < group['lookahead_steps']:
                continue

            alpha = group['slow_update_rate']
            for w_fast, w_slow in zip(group['params'], slow_weights):
                w_fast.data = alpha * w_fast.data + (1 - alpha) * w_slow.data
                w_fast.grad = None
                w_slow.data = w_fast.data

            group['step_counter'] = 0
