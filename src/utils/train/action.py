import torch
import time
from .constants import *
from collections.abc import Iterable


class Action:  # returns a single factor of loss  (   input or ...
    def __init__(
            self,
            name,
            transformed_input=None,  # can be callable ( context, loop_data, **kwargs)
            verbose=False,
            active=True,
            factor=1.,  # can be callable ( context, loop_data, **kwargs)
    ):
        self.name = name
        self.transformed_input = transformed_input
        self.verbose = verbose
        self.active = active
        self.action_factor = factor

    def toggle_verbose(self, force=None):
        self.verbose = not self.verbose if force is None else force
        return self.verbose

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        active = kwargs.get('active', self.active)
        if callable(active):
            return active(context=context, loop_data=loop_data, **kwargs)
        return active

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        action_factor = self.action_factor
        if callable(action_factor):
            action_factor = action_factor(context, loop_data, **kwargs)
        if isinstance(action_factor, (int, float)):
            return action_factor if self.is_active(context, loop_data, **kwargs) else 0.
        if context is not None:
            hparams_dict = context.get(HPARAMS_DICT, None)
            name = action_factor or f'{ACTION_FACTOR_PREFIX}{self.name}'
            if hparams_dict:
                action_factor = hparams_dict.get(name, context.get(name, 0.))
            else:
                action_factor = context.get(name, 0.)
            return action_factor if self.is_active(context, loop_data, **kwargs) else 0.
        return 1. if self.is_active(context, loop_data, **kwargs) else 0.

    def dependency_input_names(self, context=None, loop_data: dict = None, **kwargs):
        result = []
        if not self.transformed_input:
            return []
        if callable(self.transformed_input):
            result = self.transformed_input(context, loop_data, **kwargs)
        elif isinstance(self.transformed_input, Iterable):
            if isinstance(self.transformed_input, str):
                result = [self.transformed_input]
            else:
                result = list(self.transformed_input)
        if not isinstance(result, Iterable) and not result:
            return []
        if not isinstance(result, list) and result:
            return [result]
        return result

    def dependency_inputs(self, context: dict = None, loop_data: dict = None, **kwargs):
        assert loop_data is not None, f'loop_data not provided - Action: {self.name}'
        return {
            dep: loop_data.get(f'{TRANSFORM_PREFIX}{dep}', None) for dep in self.dependency_input_names(
                context, loop_data, **kwargs)
        }

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None,
               dependency_inputs: dict = None, **kwargs):
        raise NotImplemented

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None,
                 dependency_inputs: dict = None, **kwargs):
        dependency_inputs = dependency_inputs or self.dependency_inputs(context, loop_data)
        result = self.action(inputs, outputs, context, loop_data, dependency_inputs, **kwargs)
        if result is not None:
            return result

    def __repr__(self):
        if callable(self.transformed_input):
            transform = f'func({self.transformed_input.__name__})'
        elif isinstance(self.transformed_input, (list, tuple)):
            transform = '[{}]'.format(','.join(item for item in self.transformed_input))
        else:
            transform = self.transformed_input

        if callable(self.active):
            active = f'func({self.active.__name__})'
        elif isinstance(self.active, bool):
            active = str(self.active)
        else:
            active = None

        if callable(self.action_factor):
            factor = f'func({self.action_factor.__name__})'
        elif isinstance(self.action_factor, (int, float, str)):
            factor = str(self.action_factor)
        else:
            factor = None
        return '{}({})<{}{}{}{}>'.format(
            self.__class__.__name__,
            self.name,
            f'transform:{transform}, ' if transform else '',
            f'active:{active}, ' if active else '',
            f'factor:{factor}, ' if factor else '',
            'verbose' if self.verbose else '',
        )


# todo
class ContextAction:
    def __init__(self, name, active=True, verbose=False, action_function=None, trainer=None, loop=None):
        self.name = name
        self.verbose = verbose
        self.active = active
        self.action_function = action_function
        self.trainer = trainer
        self.loop = loop

    def toggle_verbose(self, force=None):
        self.verbose = not self.verbose if force is None else force
        return self.verbose

    def is_active(self, trainer=None, loop=None, **kwargs):
        trainer = trainer or self.trainer
        loop = loop or self.loop

        active = kwargs.get('active', self.active)
        if callable(active):
            return active(trainer=trainer, loop=loop, **kwargs)
        return active

    def action(self, trainer=None, loop=None, *args, **kwargs):
        trainer = trainer or self.trainer
        loop = loop or self.loop

        if callable(self.action_function):
            return self.action_function(trainer, loop, *args, **kwargs)
        raise NotImplemented

    def __call__(self, trainer=None, loop=None, *args, **kwargs):
        trainer = trainer or self.trainer
        loop = loop or self.loop
        return self.action(trainer, loop, *args, **kwargs)
