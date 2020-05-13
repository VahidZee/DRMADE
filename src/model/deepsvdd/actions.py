from src.utils.train import Action
from src.utils.train import constants
import src.config as config


class LogProb(Action):
    def __init__(self, name, transformed_input=None):
        super(LogProb, self).__init__(name, transformed_input=transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return 1.

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.transformed_input is None or self.transformed_input in dependency_inputs, \
            f'Action/{self.name} transformed input {self.transformed_input} not specified'
        if dependency_inputs:
            inputs = dependency_inputs.get(f'pgd-{self.name}', inputs)
        labels, indexes = outputs
        return - context["model"].log_prob(inputs, indexes)


class IterationDifference(Action):
    def __init__(self, name, transformed_input=None):
        super(IterationDifference, self).__init__(name, transformed_input=transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return context['hparams'].get('lambda', config.deepsvdd_lambda)

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.transformed_input is None or self.transformed_input in dependency_inputs, \
            f'Action/{self.name} transformed input {self.transformed_input} not specified'
        if dependency_inputs:
            inputs = dependency_inputs.get(f'pgd-{self.name}', inputs)
        labels, indexes = outputs
        return context["model"].iteration_difference(inputs, indexes)


class Radius(Action):
    def __init__(self, name, transformed_input=None):
        super(Radius, self).__init__(name, transformed_input=transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return 1.

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.transformed_input is None or self.transformed_input in dependency_inputs, \
            f'Action/{self.name} transformed input {self.transformed_input} not specified'
        if dependency_inputs:
            inputs = dependency_inputs.get(f'pgd-{self.name}', inputs)
        result = context["model"].radius(inputs)
        return result


class UpdateMemory(Action):
    def __init__(self, name, transformed_input=None):
        super(UpdateMemory, self).__init__(name, transformed_input=transformed_input)

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        labels, indexes = outputs
        context["model"].update_memory(loop_data["inputs"], indexes)
