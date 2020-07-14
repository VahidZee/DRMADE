from src.utils.train import Action
from src.utils.train import constants
import src.config as config
import torch


class LatentRegularization(Action):
    def __init__(self, name, function):
        super(LatentRegularization, self).__init__(
            f'latent_regularization/{name}',
            factor=f'action_factor/{name}',
            active=lambda context, loop_data, **kwargs: context[constants.HPARAMS_DICT].get(f'action_factor/{name}', 0.)
        )
        self.function = function

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return context['hparams'].get(f'{self.name}/factor', 0.)

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None, **kwargs):
        return self.function(context)(loop_data.get('false_features', loop_data.features))


latent_cor_action = LatentRegularization('correlation',
                                         lambda context: context["drmade"].encoder.latent_cor_regularization)
latent_distance_action = LatentRegularization('distance',
                                              lambda context: context["drmade"].encoder.latent_distance_regularization)
latent_zero_action = LatentRegularization('zero',
                                          lambda context: context["drmade"].encoder.latent_zero_regularization)
latent_var_action = LatentRegularization('variance',
                                         lambda context: context["drmade"].encoder.latent_var_regularization)


class EncoderAction(Action):
    def __init__(self, name='', input_transform=None, latent_transform=None, encode=False, factor=1., active=True):
        transforms = []
        if input_transform is not None:
            transforms.append(input_transform)
        if latent_transform is not None:
            transforms.append(latent_transform)
        self.latent_transform = latent_transform
        self.input_transform = input_transform
        self.encode = encode
        super(EncoderAction, self).__init__(name, transforms, factor=factor, active=active)

    def function(self, context, loop_data, inputs, latent, outputs, **kwargs):
        raise NotImplemented

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.input_transform is None or self.input_transform in dependency_inputs, \
            f'Action/{self.name} transformed input {self.input_transform} not specified in loop_data'
        assert self.latent_transform is None or self.latent_transform in dependency_inputs, \
            f'Action/{self.name} transformed input {self.input_transform} not specified in loop_data'
        inputs = dependency_inputs.get(self.input_transform, inputs)
        latent = dependency_inputs.get(
            self.latent_transform, context['drmade'].encoder(inputs) if self.encode else inputs)
        return self.function(context, loop_data, inputs, latent, outputs)


class EncoderDecoderForwardPass(EncoderAction):
    def __init__(self, name='', input_transform=None, latent_transform=None, encode=True, factor=1., active=True,
                 cross_entropy=False):
        super(EncoderDecoderForwardPass, self).__init__(
            name, input_transform, latent_transform, encode=encode, factor=factor, active=active)
        self.cross_entropy = cross_entropy

    def function(self, context, loop_data, inputs, latent, outputs, **kwargs):
        reconstruction = context['drmade'].decoder(latent)
        if not self.cross_entropy:
            return context['drmade'].decoder.distance(loop_data['inputs'], reconstruction).sum()
        return torch.nn.F.binary_cross_entropy(reconstruction, inputs, reduction='sum')


class KLDAction(EncoderAction):
    def __init__(self, name='', input_transform=None, latent_transform=None, encode=False, factor=1., active=True):
        super(KLDAction, self).__init__(
            name, input_transform, latent_transform, encode=encode, factor=factor, active=active)

    def function(self, context, loop_data, inputs, latent, outputs, **kwargs):
        latent = context['drmade'].encoder(inputs, raw=True)
        mu = latent[:, :context['drmade'].encoder.latent_size]
        log_var = latent[:, context['drmade'].encoder.latent_size:]
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


class EncoderMadeForwardPass(EncoderAction):
    def __init__(self, name='', input_transform=None, latent_transform=None, encode=True, factor=1., active=True):
        super(EncoderMadeForwardPass, self).__init__(
            name, input_transform, latent_transform, encode=encode, factor=factor, active=active)

    def function(self, context, loop_data, inputs, latent, outputs, **kwargs):
        return -context['drmade'].made.log_prob(latent)
