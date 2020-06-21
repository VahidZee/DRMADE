import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from src.models.drmade.layers import Encoder, MADE, Decoder
import src.models.drmade.config as model_config


class DRMADE(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_size,
            made=None,
            made_hidden_layers=model_config.made_hidden_layers,
            made_natural_ordering=model_config.made_natural_ordering,
            num_masks=model_config.made_num_masks,
            num_mix=model_config.num_mix,
            num_dist_parameters=model_config.num_dist_parameters,
            distribution=model_config.distribution,
            parameters_transform=model_config.parameters_transform,
            parameters_min=model_config.paramteres_min_value,
            made_use_bias=model_config.made_use_biases,
            encoder=None,
            encoder_num_layers=model_config.encoder_num_layers,
            encoder_layers_activation=model_config.encoder_layers_activation,
            encoder_latent_activation=model_config.encoder_latent_activation,
            encoder_latent_bn=model_config.encoder_bn_latent,
            encoder_use_bias=model_config.encoder_use_bias,
            encoder_generator_function=None,
            decoder=None,
            decoder_num_layers=model_config.decoder_num_layers,
            decoder_layers_activation=model_config.decoder_layers_activation,
            decoder_output_activation=model_config.decoder_output_activation,
            decoder_use_bias=model_config.decoder_use_bias,
            name=None,
            **kwargs,
    ):
        super(DRMADE, self).__init__()

        self.input_shape = input_shape
        self.latent_size = latent_size

        self.encoder = encoder or Encoder(
            input_shape=input_shape,
            latent_size=latent_size,
            num_layers=encoder_num_layers,
            layers_activation=encoder_layers_activation,
            latent_activation=encoder_latent_activation,
            bn_latent=encoder_latent_bn,
            model_generator_function=encoder_generator_function,
            bias=encoder_use_bias
        )
        self.decoder = decoder or Decoder(
            output_shape=input_shape,
            latent_size=latent_size,
            num_layers=decoder_num_layers,
            layers_activation=decoder_layers_activation,
            output_activation=decoder_output_activation,
            bias=decoder_use_bias
        )
        self.made = made or MADE(
            nin=latent_size,
            hidden_sizes=made_hidden_layers,
            num_masks=num_masks,
            natural_ordering=made_natural_ordering,
            num_dist_parameters=num_dist_parameters,
            distribution=distribution,
            parameters_transform=parameters_transform,
            parameters_min=parameters_min,
            bias=made_use_bias,
            num_mix=num_mix,
        )

        self.name = 'DRMADE:{}:{}:{}'.format(
            self.encoder.name, self.made.name, self.decoder.name
        ) if not name else name

    def forward(self, x):
        features = self.encoder(x)
        output_image = self.decoder(features)
        output = self.made(features)
        return output, features, output_image

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    @staticmethod
    def load_from_checkpoint(path=None, checkpoint=None):
        assert path is not None or checkpoint is not None, 'missing checkpoint source (path/checkpoint_dict)'
        checkpoint = checkpoint or torch.load(path)
        made = MADE.load_from_checkpoint(checkpoint=checkpoint['made_checkpoint'])
        encoder = Encoder.load_from_checkpoint(checkpoint=checkpoint['encoder_checkpoint'])
        decoder = Decoder.load_from_checkpoint(checkpoint=checkpoint['decoder_checkpoint'])
        return DRMADE(checkpoint['input_shape'], checkpoint['latent_size'], made=made, encoder=encoder, decoder=decoder,
                      name=checkpoint['name'])

    def checkpoint_dict(self):
        return {
            'input_shape': self.input_shape,
            'latent_size': self.latent_size,
            'made_checkpoint': self.made.checkpoint_dict(),
            'encoder_checkpoint': self.encoder.checkpoint_dict(),
            'decoder_checkpoint': self.decoder.checkpoint_dict(),
            'name': self.name,
        }

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), f'{path}/drmade.pth')
        torch.save(self.encoder.checkpoint_dict(), f'{path}/encoder.pth')
        torch.save(self.decoder.checkpoint_dict(), f'{path}/decoder.pth')
        torch.save(self.made.checkpoint_dict(), f'{path}/made.pth')

    def forward_ae(self, x, features=None):
        if not features:
            features = self.encoder(x)
        return self.decoder(features)

    def forward_drmade(self, x, features=None):
        if not features:
            features = self.encoder(x)
        return self.made(features)
