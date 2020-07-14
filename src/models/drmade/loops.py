from src.utils.train import Loop
from .actions import EncoderMadeForwardPass, EncoderDecoderForwardPass, KLDAction
from .input_transforms import PGDAttackAction, Encode
import src.models.drmade.config as model_config
import src.config as config


class RobustAEFeedLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, pgd_input=None, pgd_latent=None, interval=1,
                 log_interval=0, active=None, variational=False):
        pgd_input = pgd_input or dict()
        pgd_latent = pgd_latent or dict()
        input_attacker = PGDAttackAction(
            EncoderDecoderForwardPass('encoder_decoder', cross_entropy=variational),
            eps=pgd_input.get('eps', model_config.pretrain_ae_pgd_eps),
            iterations=pgd_input.get('iterations', model_config.pretrain_ae_pgd_iterations),
            randomize=pgd_input.get('randomize', model_config.pretrain_ae_pgd_randomize),
            alpha=pgd_input.get('alpha', model_config.pretrain_ae_pgd_alpha),
            input_limits=pgd_input.get('input_limits', config.input_limits),
        )
        encoder_transform = Encode('encode-pgd-input', input_attacker.name)
        latent_attacker = PGDAttackAction(
            EncoderDecoderForwardPass('decoder', encode=False, cross_entropy=variational),
            eps=pgd_latent.get('eps', model_config.pretrain_ae_latent_pgd_eps),
            iterations=pgd_latent.get('iterations', model_config.pretrain_ae_latent_pgd_iterations),
            randomize=pgd_latent.get('randomize', model_config.pretrain_ae_latent_pgd_randomize),
            alpha=pgd_latent.get('alpha', model_config.pretrain_ae_latent_pgd_alpha),
            input_limits=pgd_latent.get('input_limits', None),
            transformed_input=encoder_transform.name,
        )
        self.interval = interval

        active = active or (lambda context, *args, **kwargs:
                            self.interval and context['epoch'] % self.interval == 0)
        super(RobustAEFeedLoop, self).__init__(
            name, data_loader, device,
            input_transforms=(input_attacker, encoder_transform, latent_attacker),
            loss_actions=(
                EncoderDecoderForwardPass(
                    'reconstruction/L2' if not variational else 'reconstruction/BCE',
                    latent_transform=latent_attacker.name, encode=False, cross_entropy=variational),
                KLDAction('kld', active=variational, latent_transform=input_attacker.name, encode=False)
            ),
            optimizers=optimizers,
            log_interval=log_interval,
            active=active
        )


class RobustMadeFeedLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, pgd_input=None, pgd_latent=None, interval=1,
                 log_interval=0, active=None):
        pgd_input = pgd_input or dict()
        pgd_latent = pgd_latent or dict()
        input_attacker = PGDAttackAction(
            EncoderMadeForwardPass('encoder_made'),
            eps=pgd_input.get('eps', model_config.pretrain_encoder_made_pgd_eps),
            iterations=pgd_input.get('iterations', model_config.pretrain_encoder_made_pgd_iterations),
            randomize=pgd_input.get('randomize', model_config.pretrain_encoder_made_pgd_randomize),
            alpha=pgd_input.get('alpha', model_config.pretrain_encoder_made_pgd_alpha),
            input_limits=pgd_input.get('input_limits', config.input_limits),
        )
        encoder_transform = Encode('encode-pgd-input', input_attacker.name)
        latent_attacker = PGDAttackAction(
            EncoderMadeForwardPass('made', encode=False),
            eps=pgd_latent.get('eps', model_config.pretrain_made_pgd_eps),
            iterations=pgd_latent.get('iterations', model_config.pretrain_made_pgd_iterations),
            randomize=pgd_latent.get('randomize', model_config.pretrain_made_pgd_randomize),
            alpha=pgd_latent.get('alpha', model_config.pretrain_made_pgd_alpha),
            input_limits=pgd_latent.get('input_limits', None),
            transformed_input=encoder_transform.name,
        )
        self.interval = interval

        active = active or (lambda context, *args, **kwargs:
                            self.interval and context['epoch'] % self.interval == 0)
        super(RobustMadeFeedLoop, self).__init__(
            name, data_loader, device,
            input_transforms=(input_attacker, encoder_transform, latent_attacker),
            loss_actions=(
                EncoderMadeForwardPass('negative_logprob', latent_transform=latent_attacker.name, encode=False),
            ),
            optimizers=optimizers,
            log_interval=log_interval,
            active=active
        )
