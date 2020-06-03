from torch.optim import lr_scheduler, Adam

import src.config as config

import src.utils.train.constants as constants
import src.models.drmade.config as model_config
from src.models.drmade.loops import RobustMadeFeedLoop

from src.models.drmade.trainers.base_trainer import DRMADETrainer


class RobustEncoderMadeTrainer(DRMADETrainer):
    def __init__(self, hparams: dict = None, name=None, model=None, device=None, ):
        super().__init__(hparams, name, model, device)

        hparams = self.get(constants.HPARAMS_DICT)

        # pgd encoder made inputs
        input_limits = self.get('drmade').decoder.output_limits
        pgd_eps = hparams.get('pgd/eps', model_config.pretrain_encoder_made_pgd_eps)
        pgd_iterations = hparams.get('pgd/iterations', model_config.pretrain_encoder_made_pgd_iterations)
        pgd_alpha = hparams.get('pgd/alpha', model_config.pretrain_encoder_made_pgd_alpha)
        pgd_randomize = hparams.get('pgd/randomize', model_config.pretrain_encoder_made_pgd_randomize)
        pgd_input = {'eps': pgd_eps, 'iterations': pgd_iterations, 'alpha': pgd_alpha, 'randomize': pgd_randomize,
                     'input_limits': input_limits}
        # pgd latent
        latent_input_limits = self.get('drmade').encoder.output_limits
        pgd_latent_eps = hparams.get('pgd_latent/eps', model_config.pretrain_made_pgd_eps)
        pgd_latent_iterations = hparams.get('pgd_latent/iterations', model_config.pretrain_made_pgd_iterations)
        pgd_latent_alpha = hparams.get('pgd_latent/alpha', model_config.pretrain_made_pgd_alpha)
        pgd_latent_randomize = hparams.get('pgd_latent/randomize', model_config.pretrain_made_pgd_randomize)
        pgd_latent = {'eps': pgd_latent_eps, 'iterations': pgd_latent_iterations, 'alpha': pgd_latent_alpha,
                      'randomize': pgd_latent_randomize, 'input_limits': latent_input_limits}

        lr_decay = hparams.get('lr_decay', model_config.lr_decay)
        lr_schedule = hparams.get('lr_schedule', model_config.lr_schedule)

        # freezing unnecessary model layers
        print('freezing decoder')
        for parameter in self.get('drmade').decoder.parameters():
            parameter.requires_grad = False

        print('unfreezing encoder')
        for parameter in self.get('drmade').encoder.parameters():
            parameter.requires_grad = True

        freeze_encoder = hparams.get('freeze_encoder', False)
        made_only = False
        freeze_encoder_name = ''
        if isinstance(freeze_encoder, bool) and freeze_encoder:
            made_only = True
            print('freezing encoder')
            freeze_encoder_name = 'freezed'

            # turning off unnecessary evaluational functions
            hparams['track_extreme_reconstructions'] = hparams.get('track_extreme_reconstructions', 0)
            hparams['embedding_interval'] = hparams.get('embedding_interval', 0)
            hparams['submit_latent_interval'] = hparams.get('submit_latent_interval', 0)
            hparams['track_jacobian_interval'] = hparams.get('track_jacobian_interval', 0)

            for parameter in self.get('drmade').encoder.parameters():
                parameter.requires_grad = False

        if isinstance(freeze_encoder, dict):
            print(freeze_encoder)
            made_only = False
            conv = freeze_encoder.get('conv', tuple())
            batch_norm = freeze_encoder.get('batch_norm', conv)
            if 'batch_norm' in freeze_encoder:
                freeze_encoder_name = '{}{}'.format(
                    'conv[{}]'.format(','.join(str(i) for i in conv)) if conv else '',
                    'bn[{}]'.format(','.join(str(i) for i in batch_norm))) if batch_norm else ''
            else:
                freeze_encoder_name = 'layer[{}]'.format(','.join(str(i) for i in conv)) if conv else ''
            fc = freeze_encoder.get('fc', False)
            fc_bn = freeze_encoder.get('fc_bn', fc)
            freeze_encoder_name = '{}{}{}'.format(
                freeze_encoder_name, 'fc' if fc else '',
                'fc_bn' if fc_bn and self.get('drmade').encoder.bn_latent else '')

            print('freezing encoder')
            if conv:
                print('\tconv layer: ', end='')
                for i in conv:
                    print(i, end=' ')
                    for parameter in self.get('drmade').encoder.conv_layers[i].parameters():
                        parameter.requires_grad = False
                print()
            if batch_norm:
                print('\tbatch_norm: ', end='')
                for i in batch_norm:
                    print(i, end=' ')
                    for parameter in self.get('drmade').encoder.batch_norms[i].parameters():
                        parameter.requires_grad = False
                print()
            if fc:
                print('\tlatent')
                for parameter in self.get('drmade').encoder.fc1.parameters():
                    parameter.requires_grad = False
            if fc_bn and self.get('drmade').encoder.bn_latent:
                print('\tlatent bn')
                for parameter in self.get('drmade').encoder.latent_bn.parameters():
                    parameter.requires_grad = False

        print('unfreezing made')
        for parameter in self.get('drmade').made.parameters():
            parameter.requires_grad = True

        # optimizers and schedulers
        def lr_multiplicative_function(epoch):
            return 0.5 if lr_schedule and ((epoch + 1) % lr_schedule) == 0 else lr_decay

        self.set('lr_multiplicative_factor_lambda', lr_multiplicative_function)

        print(f'initializing learning rate scheduler - lr_decay:{lr_decay} schedule:{lr_schedule}')
        optimizer = hparams.get('optimizer', Adam)
        optimizer_hparams = hparams.get('optimizer_hparams', {'lr': model_config.base_lr, })
        print(f'initializing optimizer {optimizer.__name__} -',
              ",".join(f"{i}:{str(j)}" for i, j in optimizer_hparams.items()))

        made_optimizer = optimizer(self.get('drmade').made.parameters(), **optimizer_hparams)

        self.add_optimizer('made', made_optimizer)
        made_scheduler = lr_scheduler.MultiplicativeLR(
            made_optimizer, lr_lambda=lr_multiplicative_function, last_epoch=-1)
        self.add_scheduler('made', made_scheduler)

        if not made_only:
            encoder_optimizer = optimizer(self.get('drmade').encoder.parameters(), **optimizer_hparams)
            self.add_optimizer('encoder', encoder_optimizer)
            encoder_scheduler = lr_scheduler.MultiplicativeLR(
                encoder_optimizer, lr_lambda=lr_multiplicative_function, last_epoch=-1)
            self.add_scheduler('encoder', encoder_scheduler)

        # iterative training
        iterative = hparams.get('iterative', False)
        assert not iterative or (iterative and not made_only), \
            'cannot perform iterative training with fixed encoder'

        self.set(constants.TRAINER_NAME, name or '{}-{}{}:{}|{}{}{}{}{}|schedule{}-decay{}'.format(
            self.get(constants.TRAINER_NAME), self.get('drmade').encoder.name,
            f'({freeze_encoder_name})' if freeze_encoder_name else '', self.get('drmade').made.name,
            '' if not pgd_eps else 'pgd-eps{}-iterations{}alpha{}{}|'.format(
                pgd_eps, pgd_iterations, pgd_alpha, 'randomized' if pgd_randomize else '', ),
            '' if not pgd_latent_eps else 'pgd-latent-eps{}-iterations{}alpha{}{}|'.format(
                pgd_latent_eps, pgd_latent_iterations, pgd_latent_alpha, 'randomized' if pgd_randomize else '', ),
            'iterative|' if iterative else '',
            optimizer.__name__, '-{}'.format('-'.join(f"{i}{j}" for i, j in optimizer_hparams.items())),
            lr_schedule, lr_decay, ), replace=True)
        print("Trainer: ", self.get(constants.TRAINER_NAME))

        self.add_loop(RobustMadeFeedLoop(
            name='train-made' if iterative else 'train',
            data_loader=self.get('train_loader'),
            device=self.get(constants.DEVICE),
            optimizers=('made',) if iterative or made_only else ('made', 'encoder'),
            pgd_input=pgd_input,
            pgd_latent=pgd_latent,
            log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval)))
        if iterative:
            self.add_loop(RobustMadeFeedLoop(
                name='train-encoder',
                data_loader=self.get('train_loader'),
                device=self.get(constants.DEVICE),
                optimizers=('encoder',),
                pgd_input=pgd_input,
                pgd_latent=pgd_latent,
                log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval)))
        self.add_loop(RobustMadeFeedLoop(
            name='validation',
            data_loader=self.get('validation_loader'),
            device=self.get(constants.DEVICE),
            optimizers=tuple(),
            pgd_input=pgd_input,
            pgd_latent=pgd_latent,
            interval=hparams.get('validation_interval', model_config.validation_interval),
            log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval)))
