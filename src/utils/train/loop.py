from .constants import *
import torch
import time



class Loop:
    def __init__(
            self,
            name: str = '',
            data_loader=None,
            device=None,
            loss_actions=tuple(),  # a list of LossActions
            after_optimization_context_actions=tuple(),  # a list of ContextActions
            before_optimization_context_actions=tuple(),
            input_transforms=tuple(),
            optimizers=tuple(),  # a list of optimizer names
            log_interval=0,
            no_grad=False,
            active=True,
    ):
        self.name = name
        self.device = device
        self.optimizers = optimizers or tuple()
        self.data_loader = data_loader
        self.input_transforms = input_transforms
        self.before_optimization_context_actions = before_optimization_context_actions
        self.after_optimization_context_actions = after_optimization_context_actions
        self.loss_actions = loss_actions

        self.log_interval = log_interval
        self.no_grad = no_grad
        self.active = active
        self.loop_data = dict()
        self.verbose = False

    def is_active(self, context: dict = None, **kwargs):
        active = kwargs.get('active', self.active)
        if callable(active):
            return active(context=context, loop_data=self.loop_data, **kwargs)
        return active

    def no_grad_active(self, context=None, **kwargs):
        no_grad = kwargs.get('no_grad', self.no_grad)
        if callable(no_grad):
            return no_grad(context=context, loop_data=self.loop_data, **kwargs)
        return no_grad

    def submit_loop_data(self, context):
        for item in self.loop_data:
            if item.startswith(f'{RESULT_PREFIX}{SCALAR_PREFIX}'):
                context["writer"].add_scalar(
                    f'{item[len(f"{RESULT_PREFIX}{SCALAR_PREFIX}"):]}/{self.name}',
                    self.loop_data[item], context[EPOCH]
                )
            if item.startswith(f'{ACTION_FACTOR_PREFIX}'):
                context["writer"].add_scalar(
                    f'{item}/{self.name}',
                    self.loop_data[item], context[EPOCH]
                )

    def toggle_verbose(self, force=None):
        self.verbose = not self.verbose if force is None else force
        for action in self.loss_actions + \
                      self.before_optimization_context_actions + \
                      self.after_optimization_context_actions + \
                      self.input_transforms:
            action.toggle_verbose(force)
        return self.verbose

    def __call__(
            self,
            context=None,
            data_loader=None,
            model=None,
            device=None,
            **kwargs
    ):
        if self.verbose:
            print(f'\t- loop: {self.name} - setting up loop_data context')
        loop_data = self.loop_data
        loop_data.clear()
        assert (data_loader is None and self.data_loader is not None) or data_loader is not None, \
            f'no data_loader specified - loop:{self.name}'
        data_loader = self.data_loader or data_loader
        assert (device is None and self.device is not None) or device is not None or (
                DEVICE in context and context[DEVICE]), f'no device specified - loop:{self.name}'
        device = self.device or device
        device = device or context[DEVICE]

        time_ = time.time()

        if self.verbose:
            print(f'\t- setting up loss_action scalars, call_counts, factors')
        if self.loss_actions:
            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] = 0.

        for action in self.loss_actions:
            loop_data[f'{ACTION_FACTOR_PREFIX}{action.name}'] = 0.
            loop_data[f'{ACTION_PREFIX}{action.name}/calls_count'] = 0
            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}{action.name}'] = 0.

        with torch.set_grad_enabled(not self.no_grad_active(context, **kwargs)):
            for batch_idx, (inputs, outputs) in enumerate(data_loader):
                if self.verbose:
                    print(f'\tbidx:{batch_idx + 1}/{len(data_loader)} - converting inputs to {device}')
                inputs = inputs.to(device)
                loop_data['inputs'] = inputs
                loop_data['outputs'] = outputs

                for input_tranform in self.input_transforms:
                    if self.verbose:
                        print(
                            f'\t\t* calling transform {input_tranform.name} | active:{input_tranform.is_active(context, loop_data, **kwargs)}')
                    results = input_tranform(
                        inputs, outputs, context, loop_data=loop_data, **kwargs)
                    if isinstance(results, dict):
                        if self.verbose:
                            print(
                                f'\t\t\t- result is a dict of {list(results.keys())}')
                        for name, value in results.items():
                            loop_data[f'{TRANSFORM_PREFIX}{input_tranform.name}{name}'] = value
                    else:
                        loop_data[f'{TRANSFORM_PREFIX}{input_tranform.name}'] = results

                loss = torch.zeros(1).to(device)
                for action in self.loss_actions:
                    factor = action.factor(context, loop_data=loop_data, **kwargs)
                    active = action.is_active(context, loop_data, **kwargs)
                    if self.verbose:
                        print(
                            f'\t\t* calling loss_action {action.name} | active:{active} - factor: {factor} - term_loss:',
                            end='')
                    if active:
                        loop_data[f'{ACTION_PREFIX}{action.name}/calls_count'] += 1
                        loop_data[f'{ACTION_FACTOR_PREFIX}{action.name}'] += factor
                        result = action(
                            inputs, outputs, context, loop_data=loop_data, **kwargs)
                        if factor:
                            loss += factor * result
                        if self.verbose:
                            print(
                                f'{result} - total_loss:{loss}')
                        loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}{action.name}'] += result.data / inputs.shape[0]

                loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] += loss.data / inputs.shape[0]

                for action in self.before_optimization_context_actions:
                    if action.is_active(context, loop_data=loop_data, **kwargs):
                        action(inputs, outputs, context, loop_data=loop_data, **kwargs)

                if self.optimizers and self.loss_actions:
                    for optimizer_name in self.optimizers:
                        context[f'{OPTIMIZER_PREFIX}{optimizer_name}'].zero_grad()
                    loss.backward()
                    for optimizer_name in self.optimizers:
                        context[f'{OPTIMIZER_PREFIX}{optimizer_name}'].step()

                for action in self.after_optimization_context_actions:
                    if action.is_active(context, loop_data=loop_data, **kwargs):
                        action(inputs, outputs, context, loop_data=loop_data, **kwargs)

                if self.log_interval and (batch_idx + 1) % self.log_interval == 0:
                    print(
                        '\t{}\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                            self.name,
                            batch_idx + 1,
                            len(data_loader),
                            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'].item() / (1 + batch_idx),
                            time.time() - time_))
                    time_ = time.time()

        for item in loop_data:
            if item.startswith(f'{RESULT_PREFIX}{SCALAR_PREFIX}'):
                loop_data[item] /= len(data_loader)
            if item.startswith(f'{ACTION_FACTOR_PREFIX}'):
                loop_data[item] /= loop_data[f'{ACTION_PREFIX}{item[len(ACTION_FACTOR_PREFIX):]}/calls_count']
        self.loop_data = loop_data
        return loop_data

    def __repr__(self):
        if callable(self.active):
            active = f'\t+ Active: func({self.active.__name__})\n'
        elif isinstance(self.active, bool):
            active = f'\t+ Active: {str(self.active)}\n'
        else:
            active = ''

        if callable(self.no_grad):
            no_grad = f'\t+ No-Gradient: func({self.no_grad.__name__})\n'
        elif isinstance(self.active, bool):
            no_grad = f'\t+ No-Gradient: {str(self.no_grad)}\n'
        else:
            no_grad = ''

        device = f'\t+ Device: {self.device}\n'
        data_loader = f'\t+ Data Loader: {self.data_loader}\n' if self.data_loader else ''
        log_interval = f'\t+ Log Interval: {self.log_interval}\n' if self.log_interval else ''
        optimizers = '\t+ Optimizers: [ {} ]\n'.format(
            ', '.join(optimizer for optimizer in self.optimizers)) if self.optimizers else ''
        verbose = '\t+ Verbose\n' if self.verbose else ''
        input_transforms = '\t+ Input Transforms:\n\t\t- {}\n'.format(
            '\n\t\t- '.join(repr(transform) for transform in self.input_transforms),
        ) if self.input_transforms else ''
        loss_actions = '\t+ Loss Actions:\n\t\t- {}\n'.format(
            '\n\t\t- '.join(repr(action) for action in self.loss_actions),
        ) if self.loss_actions else ''
        before_op_context_actions = '\t+ Before Optimization Context Actions:\n\t\t- {}\n'.format(
            '\n\t\t- '.join(repr(action) for action in self.before_optimization_context_actions),
        ) if self.before_optimization_context_actions else ''
        after_op_context_actions = '\t+ After Optimization Context Actions:\n\t\t- {}\n'.format(
            '\n\t\t- '.join(repr(action) for action in self.before_optimization_context_actions),
        ) if self.after_optimization_context_actions else ''
        return 'Loop {}({})\n{}{}{}{}{}{}{}{}{}{}{}'.format(
            self.__class__.__name__, self.name, device, data_loader, active, no_grad, input_transforms,
            loss_actions, before_op_context_actions, optimizers, after_op_context_actions, log_interval, verbose)
