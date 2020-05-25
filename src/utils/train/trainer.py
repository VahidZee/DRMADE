from .constants import *

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class Trainer:
    def __init__(
            self,
            name: str = '',
            context: dict = None,
            loops=tuple(),
            schedulers: dict = None,
            optimizers: dict = None,
            verbose: bool = False,
    ):
        self.context = context or dict()
        if EPOCH not in self.context:
            self.context[EPOCH] = 0

        if TRAINER_NAME not in self.context:
            self.context[TRAINER_NAME] = name

        self.loops_list = self.context[LOOPS_LIST] = list()
        for loop in loops:
            self.add_loop(loop)

        self.optimizers_list = self.context[OPTIMIZERS_LIST] = list()
        if optimizers:
            for name, optimizer in optimizers.items():
                self.add_optimizer(name, optimizer)

        self.schedulers_list = self.context[SCHEDULERS_LIST] = list()
        if schedulers:
            for name, scheduler in schedulers.items():
                self.add_scheduler(name, scheduler)
        self.verbose = verbose

    def add_loop(self, loop, replace=False):
        index = self.context.get(f'{LOOP_PREFIX}{loop.name}/index', None)
        assert replace or index is None, \
            f'Trainer {self.get(TRAINER_NAME)} - loop "{loop.name}" already exists'
        self.set(f'{LOOP_PREFIX}{loop.name}', loop, replace)
        self.set(f'{LOOP_PREFIX}{loop.name}/data', None, replace)
        if index is not None:
            self.loops_list[index] = loop
        else:
            self.set(f'{LOOP_PREFIX}{loop.name}/index', len(self.loops_list))
            self.loops_list.append(loop)

    def add_optimizer(self, name, optimizer, replace=False):
        index = self.context.get(f'{OPTIMIZER_PREFIX}{name}/index', None)
        assert replace or index is None, \
            f'Trainer {self.get(TRAINER_NAME)} - optimizer "{name}" already exists'
        self.set(f'{OPTIMIZER_PREFIX}{name}', optimizer, replace)
        if index is not None:
            self.optimizers_list[index] = optimizer
        else:
            self.set(f'{OPTIMIZER_PREFIX}{name}/index', len(self.optimizers_list))
            self.set(f'{OPTIMIZER_PREFIX}index/{len(self.optimizers_list)}', name)
            self.optimizers_list.append(optimizer)

    def add_scheduler(self, name, scheduler, replace=False):
        index = self.context.get(f'{SCHEDULER_PREFIX}{name}/index', None)
        assert replace or index is None, \
            f'Trainer {self.get(TRAINER_NAME)} - scheduler "{name}" already exists'
        self.set(f'{SCHEDULER_PREFIX}{name}', scheduler, replace)
        if index is not None:
            self.optimizers_list[index] = scheduler
        else:
            self.set(f'{SCHEDULER_PREFIX}{name}/index', len(self.schedulers_list))
            self.set(f'{SCHEDULER_PREFIX}index/{len(self.schedulers_list)}', name)
            self.schedulers_list.append(scheduler)

    def get(self, item, prefix=None, **kwargs):
        if 'default' in kwargs:
            return self.context.get(item if prefix is None else f'{prefix}{item}', kwargs['default'])
        if not prefix:
            assert item in self.context, f'Trainer {self.get(TRAINER_NAME)} - context does not have "{item}"'
            return self.context[item]
        assert item in self.context, f'Trainer {self.get(TRAINER_NAME)} - context does not have "{prefix}{item}"'
        return self.context[f'{prefix}{item}']

    def set(self, name, value, replace=False):
        assert replace or f'{name}' not in self.context, \
            f'Trainer {self.get(TRAINER_NAME)} - "{name}" already exists in context'
        self.context[name] = value

    def submit_progress(self):
        for index, optimizer in enumerate(self.optimizers_list):
            self.get("writer").add_scalar(
                f'{LEARNING_RATE_PREFIX}{self.get(f"{OPTIMIZER_PREFIX}index/{index}")}',
                optimizer.param_groups[0]["lr"], self.context[EPOCH]
            )
            print('\t+ optimizer', self.get(f"{OPTIMIZER_PREFIX}index/{index}"), '- lr:',
                  optimizer.param_groups[0]["lr"])

    def stop_training(self):
        raise NotImplemented

    def toggle_verbose(self, force=None):
        self.verbose = not self.verbose if force is None else force
        for loop in self.loops_list:
            loop.toggle_verbose(self.verbose)
        return self.verbose

    def train(self):
        while not self.stop_training():
            print(f'epoch {self.context[EPOCH]:5d}')
            for loop in self.context['loops']:
                if loop.is_active(self.context):
                    self.set(f'{LOOP_PREFIX}{loop.name}/data', loop(self.context), replace=True)
                    loop.submit_loop_data(self.context)
            self.submit_progress()
            self.context[EPOCH] += 1

    def __repr__(self):
        name = f'Trainer {self.__class__.__name__}({self.get(TRAINER_NAME)})\n'
        loops = '- Loops({}):\n* {}\n'.format(
            len(self.loops_list), '\n* '.join(repr(loop) for loop in self.loops_list)) if self.loops_list else ''
        optimizers = '- Optimizers({}): [ {} ]\n'.format(
            len(self.optimizers_list),
            ', '.join(self.get(f'{OPTIMIZER_PREFIX}index/{i}') for i in
                      range(len(self.optimizers_list)))) if self.optimizers_list else ''
        schedulers = '- Schedulers({}): [ {} ]\n'.format(
            len(self.schedulers_list),
            ', '.join(self.get(f'{SCHEDULER_PREFIX}index/{i}') for i in
                      range(len(self.schedulers_list)))) if self.schedulers_list else ''
        verbose = '- Verbose\n' if self.verbose else ''
        return '{}{}{}{}{}'.format(name, optimizers, schedulers,verbose, loops)
