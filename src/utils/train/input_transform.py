import torch
import time
from collections.abc import Iterable
from .constants import *


class InputTransform:
    def __init__(
            self,
            name,
            transformed_input=None,  # can be callable ( context, loop_data, **kwargs)
            active=True,  # can be callable (context, loop_data, **kwargs)
            verbose=False,
            direct=True,
    ):
        self.name = name
        self.transformed_input = transformed_input
        self.active = active
        self.verbose = verbose
        self.direct = direct

    def toggle_verbose(self, force=None):
        self.verbose = not self.verbose if force is None else force
        return self.verbose

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        active = kwargs.get('active', self.active)
        if callable(active):
            return active(context=context, loop_data=loop_data, **kwargs)
        return active

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
        if not isinstance(result, list) and result:
            return [result]
        return result

    def dependency_inputs(self, context: dict = None, loop_data: dict = None, **kwargs):
        assert loop_data is not None, f'loop_data not provided - InputTransform: {self.name}'
        return {
            dep: loop_data.get(f'{TRANSFORM_PREFIX}{dep}', None) for dep in self.dependency_input_names(
                context, loop_data, **kwargs)
        }

    def changes_input_directly(self, context=None, loop_data: dict = None, **kwargs):
        return self.direct

    def transform(self, inputs, outputs=None, context=None, loop_data: dict = None,
                  dependency_inputs: dict = None, **kwargs):
        raise NotImplemented

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None,
                 dependency_inputs: dict = None, **kwargs):
        dependency_inputs = dependency_inputs or self.dependency_inputs(context, loop_data)
        if self.is_active(context, loop_data, **kwargs):
            return self.transform(inputs, outputs, context, loop_data, dependency_inputs, **kwargs)
        else:
            return inputs

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

        return '{}({})<{}{}{}{}>'.format(
            self.__class__.__name__,
            self.name,
            f'transform:{transform}, ' if transform else '',
            f'active:{active}, ' if active else '',
            'verbose, ' if self.verbose else '',
            'direct' if self.direct else ''
        )
