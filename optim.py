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


class AdamW(Optimizer):
    """AdamW implemention from Pytorch"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
