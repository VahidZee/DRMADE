import numpy as np

import torch
import torch.nn as nn

import src.models.drmade.config as model_config
from src.models.drmade.layers.utility_layers import View
from collections.abc import Iterable
import dill


class Encoder(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_size,
            num_layers=1,
            name=None,
            model_generator_function=None,
            bias=model_config.encoder_use_bias,
            bn_affine=model_config.encoder_bn_affine,
            bn_eps=model_config.encoder_bn_eps,
            bn_latent=model_config.encoder_bn_latent,
            layers_activation=model_config.encoder_layers_activation,
            latent_activation=model_config.encoder_latent_activation,
            variational=False,
            **kwargs
    ):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_input_channels = input_shape[0]
        self.latent_size = latent_size
        self.bn_latent = bn_latent
        self.latent_activation = latent_activation
        self.layers_activation = layers_activation
        self.input_size = min(input_shape[1], input_shape[2])
        self.bias = bias
        self.bn_affine = bn_affine
        self.bn_eps = bn_eps
        self.variational = variational

        self.name = name or '{}Encoder{}{}{}-{}{}{}{}'.format(
            'var' if self.variational else '',
            self.num_layers,
            self.layers_activation,
            'bn_affine' if bn_affine else '',
            self.latent_size,
            self.latent_activation,
            'bn' if self.bn_latent else '',
            '(bias)' if self.bias else '',
        ) if not name else name
        self.model_generator_function = model_generator_function

        self.output_limits = None
        self.activate_latent = None
        if model_generator_function is not None:
            self.model, name = model_generator_function(self)
            self.name = name or self.name
            return
        self.model = nn.Sequential()

        assert num_layers > 0, 'non-positive number of layers'
        latent_image_size = self.input_size // (2 ** num_layers)

        assert latent_image_size > 0, 'number of layers is too large'
        assert latent_activation in ['', 'tanh', 'leaky_relu', 'sigmoid'], 'unknown latent activation'
        assert layers_activation in ['relu', 'elu', 'leaky_relu'], 'non-positive number of layers'

        # convolutional layers
        self.conv_layers = []
        for i in range(self.num_layers):
            layer = nn.Sequential()
            conv = nn.Conv2d(32 * (2 ** (i - 1)) if i else self.num_input_channels, 32 * (2 ** i), 5, bias=bias,
                             padding=2)
            if self.layers_activation == 'elu':
                nn.init.xavier_uniform_(conv.weight)
            else:
                nn.init.xavier_uniform_(conv.weight, nn.init.calculate_gain(self.layers_activation))
            layer.add_module('conv', conv)
            layer.add_module('batch_norm', nn.BatchNorm2d(32 * (2 ** i), eps=bn_eps, affine=bn_affine))

            activation = None
            if self.layers_activation == 'leaky_relu':
                activation = nn.LeakyReLU()
            if self.layers_activation == 'elu':
                activation = nn.ELU()
            if self.layers_activation == 'relu':
                activation = nn.ReLU()
            layer.add_module('activation', activation)
            layer.add_module('pool', nn.MaxPool2d(2, 2))
            self.conv_layers.append(layer)

            self.model.add_module(f'layer{i}', layer)

        self.model.add_module('flatten', View())

        # fully connected layer
        self.fc = []
        fc_block = nn.Sequential()
        fc = nn.Linear(32 * (2 ** (num_layers - 1)) * (latent_image_size ** 2),
                       self.latent_size * 2 if self.variational else self.latent_size, bias=bias)
        if not self.latent_activation and self.latent_activation:
            nn.init.xavier_uniform_(fc.weight, nn.init.calculate_gain(self.latent_activation))
        self.fc.append(fc)
        fc_block.add_module('fc', fc)

        # fully connected activation
        if self.latent_activation == 'tanh':
            self.output_limits = (-1, 1.)
            self.activate_latent = torch.nn.Tanh()
        elif self.activate_latent == 'leaky_relu':
            self.activate_latent = torch.nn.LeakyReLU()
        elif self.activate_latent == 'sigmoid':
            self.activate_latent = torch.nn.Sigmoid()
            self.output_limits = (0.,)
        if self.activate_latent is not None:
            self.fc.append(self.activate_latent)
            fc_block.add_module('activation', self.activate_latent)

        # fully connected batch_norm
        self.latent_bn = nn.BatchNorm1d(self.latent_size, eps=bn_eps, affine=bn_affine) if self.bn_latent else None
        if self.bn_latent:
            self.fc.append(self.latent_bn)
            fc_block.add_module('batch_norm', self.latent_bn)
        self.model.add_module('fc', fc_block)

    def forward(self, x: torch.Tensor, raw=False):
        latent = self.model(x)
        if not self.variational or raw:
            return latent
        else:
            # first half of latent vector is mu, second half is logvar
            return torch.exp(0.5 * latent[:, self.latent_size:]) * torch.rand(
                x.shape[0], self.latent_size, device=x.device) + latent[:, :self.latent_size]

    def latent_cor_regularization(self, features):
        norm_features = features / ((features ** 2).sum(1, keepdim=True) ** 0.5).repeat(1, self.latent_size)
        correlations = norm_features @ norm_features.reshape(self.latent_size, -1)
        if model_config.latent_cor_regularization_abs:
            return (torch.abs(correlations)).sum()
        return correlations.sum()

    def latent_distance_regularization(
            self, features, use_norm=model_config.latent_distance_normalize_features,
            norm=model_config.latent_distance_norm
    ):
        batch_size = features.shape[0]
        vec = features
        if use_norm:
            vec = features / ((features ** norm).sum(1, keepdim=True) ** (1 / norm)).repeat(1, self.latent_size)
        a = vec.repeat(1, batch_size).reshape(-1, batch_size, self.latent_size)
        b = vec.repeat(batch_size, 1).reshape(-1, batch_size, self.latent_size)
        return (1 / ((torch.abs(a - b) ** norm + 1).sum(2) ** (1 / norm))).sum()

    def latent_zero_regularization(self, features, eps=model_config.latent_zero_regularization_eps):
        return torch.sum(1.0 / (eps + torch.abs(features)))

    def latent_var_regularization(self, features):
        return torch.sum(((features - features.sum(1, keepdim=True) / self.latent_size) ** 2).sum(1) / self.latent_size)

    @staticmethod
    def load_from_checkpoint(path=None, checkpoint=None):
        assert path is not None or checkpoint is not None, 'missing checkpoint source (path/checkpoint_dict)'
        checkpoint = checkpoint or torch.load(path)
        state_dict = checkpoint['state_dict']
        checkpoint['model_generator_function'] = dill.loads(checkpoint['model_generator_function'])
        instance = Encoder(**checkpoint)
        instance.load_state_dict(state_dict, strict=True)
        print('encoder loaded from', path or 'checkpoint_dict')
        return instance

    def checkpoint_dict(self):
        return {
            'state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'bias': self.bias,
            'bn_affine': self.bn_affine,
            'bn_eps': self.bn_eps,
            'model_generator_function': dill.dumps(self.model_generator_function),
            'layers_activation': self.layers_activation,
            'latent_activation': self.latent_activation,
            'name': self.name,
            'num_layers': self.num_layers,
            'latent_size': self.latent_size,
            'bn_latent': self.bn_latent,
            'variational': self.variational,
        }

    def freeze(self, instruction, verbose=True):
        if isinstance(instruction, bool) and instruction:
            if verbose:
                print('freezing all of encoder')
            for par in self.parameters():
                par.requires_grad = False
            return True, 'freezed'

        # todo: O(n2) could be optimized
        if isinstance(instruction, dict) and instruction:
            if verbose:
                print('freezing encoder')
            freeze_pars = []
            state = 'freezed[{}]'.format(','.join(
                f'{key}' if item is None else '{}[{}]'.format(key, ",".join(str(i) for i in item)) for key, item in
                instruction.items()))

            for key, item in instruction.items():
                if isinstance(item, Iterable):
                    for value in item:
                        freeze_pars.append(f'{key}{value}')
                else:
                    freeze_pars.append(str(key))
            pars_count = counter = 0
            for phrase in freeze_pars:
                pars_count = 0
                for name, par in self.named_parameters():
                    pars_count += 1
                    if phrase in name:
                        if verbose:
                            print('\tfroze', phrase, '-', name)
                        par.requires_grad = False
                        counter += 1
            return counter == pars_count, state
        return False, ''
