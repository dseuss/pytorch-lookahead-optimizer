from collections import defaultdict
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

        self.fast_optimizer = fast_optimizer(params, **kwargs)

        # TODO Proper handling of params for fast optimizer
        self.defaults = dict(
            slow_update_rate=slow_update_rate,
            lookahead_steps=lookahead_steps,
            step_counter=0)
        self.state = defaultdict(dict)

        for group in self.fast_optimizer.param_groups:
            for name, value in self.defaults.items():
                group[name] = group.get(name, value)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.fast_optimizer.step()

        for group in self.param_groups:
            group['step_counter'] += 1
            if group['step_counter'] < group['lookahead_steps']:
                continue

            alpha = group['slow_update_rate']
            for tensor in group['params']:
                state = self.state[tensor]
                if len(state) == 0:
                    state['slow_weight'] = tensor.clone().detach()
                    state['slow_weight'].requires_grad = False

                slow_weight = state['slow_weight']
                tensor.data = alpha * tensor.data + (1 - alpha) * slow_weight
                tensor.grad = None
                slow_weight.data = tensor.data

            group['step_counter'] = 0

    def add_param_group(self, group):
        self.fast_optimizer.add_param_group(group)
        group = self.fast_optimizer.param_groups[-1]
        for name, value in self.defaults.items():
            group[name] = group.get(name, value)

    @property
    def param_groups(self):
        return self.fast_optimizer.param_groups
