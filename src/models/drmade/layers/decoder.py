import numpy as np

import torch
import torch.nn as nn

import src.models.drmade.config as model_config
from .utility_layers import Interpolate, View
import dill


class Decoder(nn.Module):
    def __init__(
            self,
            output_shape,
            latent_size,
            num_layers,
            layers_activation=model_config.decoder_layers_activation,
            output_activation=model_config.decoder_output_activation,
            bias=model_config.decoder_use_bias,
            bn_affine=model_config.decoder_bn_affine,
            bn_eps=model_config.decoder_bn_eps,
            name=None,
            **kwargs
    ):
        super().__init__()
        # Decoder
        num_layers = int(num_layers)
        assert num_layers > 0, 'non-positive number of layers'

        self.latent_image_size = min(output_shape[1], output_shape[2]) // (2 ** num_layers)
        assert self.latent_image_size > 0, 'number of layers is too large'

        assert output_activation in ['tanh', 'sigmoid'], 'unknown output activation function'
        assert layers_activation in ['relu', 'elu', 'leaky_relu'], 'unknown layers activation function'

        self.output_shape = output_shape
        self.output_size = min(output_shape[1], output_shape[2])
        self.output_activation = output_activation
        self.layers_activation = layers_activation
        self.num_output_channels = output_shape[0]
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bn_eps = bn_eps
        self.bn_affine = bn_affine
        self.bias = bias

        self.latent_num_channels = (
                latent_size // (self.latent_image_size * self.latent_image_size) + 1) if latent_size % (
                self.latent_image_size * self.latent_image_size) != 0 else (
                latent_size // (self.latent_image_size * self.latent_image_size))
        self.name = 'Decoder{}{}{}-{}'.format(
            self.num_layers,
            self.layers_activation,
            'bn_affine' if bn_affine else '',
            self.output_activation,
        ) if not name else name

        self.model = nn.Sequential()
        if latent_size % (self.latent_image_size * self.latent_image_size) != 0:
            self.model.add_module(
                'latent_transform',
                nn.Linear(latent_size, self.latent_num_channels * self.latent_image_size * self.latent_image_size,
                          bias=bias))
        self.model.add_module('view', View((self.latent_num_channels, self.latent_image_size, self.latent_image_size)))
        self.deconv_layers = []
        self.output_limits = None
        last_size = self.latent_image_size
        for i in range(self.num_layers + 1):
            layer = nn.Sequential()
            n_input_channels = 2 ** (5 + self.num_layers - i) if i else self.latent_num_channels
            n_output_channels = 2 ** (4 + self.num_layers - i) if i != self.num_layers else self.num_output_channels
            kernel_size = 6 if (self.output_size // (2 ** (self.num_layers - i - 1))) == 2 * (last_size + 1) else 5
            if i == self.num_layers:
                kernel_size = 5 + self.output_size - last_size
            last_size = (last_size + 1) * 2 if (self.output_size // (2 ** (self.num_layers - i - 1))) == 2 * (
                    last_size + 1) else last_size * 2
            deconv = nn.ConvTranspose2d(n_input_channels, n_output_channels, kernel_size, bias=bias, padding=2)
            layer.add_module('deconv', deconv)

            layer.add_module('batch_norm', nn.BatchNorm2d(n_output_channels, eps=bn_eps, affine=bn_affine))
            activation = None
            if i != self.num_layers:
                if self.output_activation == 'elu':
                    nn.init.xavier_uniform_(deconv.weight)
                else:
                    nn.init.xavier_uniform_(deconv.weight, nn.init.calculate_gain(self.output_activation))
                if self.layers_activation == 'leaky_relu':
                    activation = nn.LeakyReLU()
                if self.layers_activation == 'elu':
                    nn.init.xavier_uniform_(deconv.weight)
                    activation = nn.ELU()
                if self.layers_activation == 'relu':
                    activation = nn.ReLU()
                if activation is not None:
                    layer.add_module('activation', activation)
                layer.add_module('interpolate', Interpolate(2))
            else:
                nn.init.xavier_uniform_(deconv.weight, nn.init.calculate_gain(self.output_activation))
                if self.output_activation == 'tanh':
                    self.output_limits = (-1., 1.)
                    activation = nn.Tanh()
                if self.layers_activation == 'sigmoid':
                    self.output_limits = (0., 1.)
                    activation = nn.Sigmoid()
                layer.add_module('activation', activation)
            self.deconv_layers.append(deconv)
            self.model.add_module(f'layer{i}', layer)

    def forward(self, x):
        return self.model(x)

    def distance_hitmap(self, input_image, output_image):
        return torch.abs(input_image - output_image)

    def distance(self, input_image, output_image, norm=2):
        return ((self.distance_hitmap(input_image, output_image) ** norm).sum(1).sum(1).sum(
            1) + model_config.decoder_distance_eps) ** (1 / norm)

    @staticmethod
    def load_from_checkpoint(path=None, checkpoint=None):
        assert path is not None or checkpoint is not None, 'missing checkpoint source (path/checkpoint_dict)'
        checkpoint = checkpoint or torch.load(path)
        state_dict = checkpoint['state_dict']
        instance = Decoder(**checkpoint)
        instance.load_state_dict(state_dict, strict=True)
        print('decoder loaded from', path or 'checkpoint_dict')
        return instance

    def checkpoint_dict(self):
        return {
            'state_dict': self.state_dict(),
            'output_shape': self.output_shape,
            'latent_size': self.latent_size,
            'num_layers': self.num_layers,
            'layers_activation': self.layers_activation,
            'output_activation': self.output_activation,
            'bias': self.bias,
            'bn_affine': self.bn_affine,
            'bn_eps': self.bn_eps,
            'name': self.name,
        }