import functools as ft
from time import time


def get_log_prefix(engine):
    n_iterations = len(engine.state.dataloader)
    return (f'[Epoch {engine.state.epoch}/{engine.state.max_epochs}] '
            f'[Iteration {engine.state.iteration % n_iterations}/{n_iterations}]')


def step_lr_scheduler(optimizer, scheduler, on_epoch=True, summary_writer=None,
                      verbose=True):

    def func(engine):
        scheduler.step(engine.state.epoch if on_epoch else engine.state.iteration)
        learning_rates = [g['lr'] for g in optimizer.param_groups]
        if verbose:
            print(f'{get_log_prefix(engine)} Set learning rates to {learning_rates}')

        if summary_writer is not None:
            for n, param_group in enumerate(optimizer.param_groups):
                summary_writer.add_scalar(
                    f'stats/lr_{n}', param_group['lr'], engine.state.iteration)

    return func


def log_iterations_per_second(n=10, summary_writer=None, verbose=True):
    def func(engine):
        loader = engine.state.dataloader
        counter = (engine.state.iteration - 1) % len(loader) + 1
        if counter % n != 0:
            return

        last_called = getattr(engine.state, 'last_called', None)
        if (last_called is not None) and counter >= n:
            runtime = time() - last_called
            it_per_s = n / runtime

            if verbose:
                print(f'{get_log_prefix(engine)} {it_per_s:.2f} it/s')
            if summary_writer is not None:
                summary_writer.add_scalar(
                    f'stats/it_per_s', it_per_s, engine.state.iteration)

        engine.state.last_called = time()
    return func


def every_n(n, callback=None, on_epoch=False):
    def func(engine, callback=None):
        if on_epoch:
            counter = engine.state.epoch
        else:
            loader = engine.state.dataloader
            counter = (engine.state.iteration - 1) % len(loader) + 1

        if counter % n == 0:
            callback(engine)

    return ft.partial(func, callback=callback) if callback is not None \
        else (lambda c: ft.partial(func, callback=c))