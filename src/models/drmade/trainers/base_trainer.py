import torch as torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd.functional import jacobian
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score
from src.utils.data import DatasetSelection
from src.models.drmade.model import DRMADE, Encoder, Decoder, MADE
import src.config as config

from src.utils.train import Trainer
import src.utils.train.constants as constants
import src.models.drmade.config as model_config


class DRMADETrainer(Trainer):
    def __init__(self, hparams: dict = None, name='', drmade=None, device=None, checkpoint_path=None):
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            return
        # reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.set_default_tensor_type('torch.FloatTensor')
        context = dict()
        hparams = hparams or dict()
        context[constants.HPARAMS_DICT] = hparams
        context['normal_classes'] = hparams.get('normal_classes', config.normal_classes)

        # aquiring device cuda if available
        context[constants.DEVICE] = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", context[constants.DEVICE])

        print('loading training data')
        context['train_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=context['normal_classes'], train=True)
        print('loading validation data')
        context['validation_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=context['normal_classes'], train=False)
        print('loading test data')
        context['test_data'] = DatasetSelection(hparams.get('dataset', config.dataset), train=False)

        context['input_shape'] = context['train_data'].input_shape()

        print('initializing data loaders')
        context['train_loader'] = context['train_data'].get_dataloader(
            shuffle=True, batch_size=hparams.get('train_batch_size', config.train_batch_size))
        context['validation_loader'] = context['validation_data'].get_dataloader(
            shuffle=False, batch_size=hparams.get('validation_batch_size', config.validation_batch_size))
        context['test_loader'] = context['test_data'].get_dataloader(
            shuffle=False, batch_size=hparams.get('test_batch_size', config.test_batch_size))

        print('initializing models')
        checkpoint_encoder = hparams.get('checkpoint_encoder', model_config.checkpoint_encoder)
        encoder = hparams.get('encoder', None)
        if checkpoint_encoder:
            encoder = Encoder.load_from_checkpoint(checkpoint_encoder, hparams.get('checkpoint_encoder_dict', None))

        decoder = hparams.get('decoder', None)
        checkpoint_decoder = hparams.get('checkpoint_decoder', model_config.checkpoint_drmade)
        if checkpoint_decoder:
            decoder = Decoder.load_from_checkpoint(checkpoint_decoder, hparams.get('checkpoint_decoder_dict', None))

        made = hparams.get('made', None)
        checkpoint_made = hparams.get('checkpoint_made', model_config.checkpoint_drmade)
        if checkpoint_made:
            made = MADE.load_from_checkpoint(checkpoint_made, hparams.get('checkpoint_made_dict', None))

        drmade = hparams.get('drmade', drmade)
        checkpoint_drmade = hparams.get('checkpoint_drmade', model_config.checkpoint_drmade)
        if checkpoint_drmade:
            drmade = DRMADE.load_from_checkpoint(checkpoint_made, hparams.get('checkpoint_drmade_dict', None))

        drmade = drmade or DRMADE(
            input_shape=context["input_shape"],
            latent_size=hparams.get('latent_size', model_config.latent_size),
            made=made,
            made_hidden_layers=hparams.get('made_hidden_layers', model_config.made_hidden_layers),
            made_natural_ordering=hparams.get('made_natural_ordering', model_config.made_natural_ordering),
            num_masks=hparams.get('made_num_masks', model_config.made_num_masks),
            num_mix=hparams.get('num_mix', model_config.num_mix),
            num_dist_parameters=hparams.get('num_dist_parameters', model_config.num_dist_parameters),
            distribution=hparams.get('distribution', model_config.distribution),
            parameters_transform=hparams.get('parameters_transform', model_config.parameters_transform),
            parameters_min=hparams.get('parameters_min_value', model_config.paramteres_min_value),
            encoder=encoder,
            encoder_num_layers=hparams.get('encoder_num_layers', model_config.encoder_num_layers),
            encoder_layers_activation=hparams.get('encoder_layers_activation', model_config.encoder_layers_activation),
            encoder_latent_activation=hparams.get('encoder_latent_activation', model_config.encoder_latent_activation),
            encoder_latent_bn=hparams.get('encoder_bn_latent', model_config.encoder_bn_latent),
            encoder_generator_function=hparams.get('encoder_generator_function', None),
            decoder=decoder,
            decoder_num_layers=hparams.get('decoder_num_layers', model_config.decoder_num_layers),
            decoder_layers_activation=hparams.get('decoder_layers_activation', model_config.decoder_layers_activation),
            decoder_output_activation=hparams.get('decoder_output_activation', model_config.decoder_output_activation),
        ).to(context[constants.DEVICE])
        drmade.encoder = drmade.encoder.to(context[constants.DEVICE])
        drmade.made = drmade.made.to(context[constants.DEVICE])
        drmade.decoder = drmade.decoder.to(context[constants.DEVICE])

        print(f'models: {drmade.name} was initialized')
        # setting up context name
        name = name or '{}[{}]'.format(
            hparams.get('dataset', config.dataset).__name__,
            ','.join(str(i) for i in hparams.get('normal_classes', config.normal_classes)),
        )
        self.summarizer_ready = False
        super().__init__(name, context, models={'drmade': drmade})

    def setup_output_directories(self, output_root=None):
        self.set('output_root', output_root if output_root else self.get(constants.HPARAMS_DICT).get(
            'output_root', config.output_root))
        self.set('models_dir', f'{self.get("output_root")}/models')
        self.set('check_point_saving_dir', f'{self.get("models_dir")}/{self.get(constants.TRAINER_NAME)}')

        self.set('runs_dir', f'{self.get("output_root")}/runs')

        # ensuring the existance of output directories
        Path(self.get('output_root')).mkdir(parents=True, exist_ok=True)
        Path(self.get('models_dir')).mkdir(parents=True, exist_ok=True)
        Path(self.get('check_point_saving_dir')).mkdir(parents=True, exist_ok=True)
        Path(self.get('runs_dir')).mkdir(parents=True, exist_ok=True)

    def setup_writer(self, output_root=None):
        if self.summarizer_ready:
            return
        self.summarizer_ready = True
        self.setup_output_directories()
        self.set('writer', SummaryWriter(log_dir=f'{self.get("runs_dir")}/{self.get(constants.TRAINER_NAME)}'))

    def save_checkpoint(self, output_path=None):
        output_path = (output_path or self.get(
            'check_point_saving_dir')) + f'/{self.get(constants.TRAINER_NAME)}-E{self.get("epoch")}'
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.get('drmade').save(output_path)
        torch.save(self.checkpoint_dict(), output_path + f'/trainer_checkpoint.pth')

    def load_checkpoint(self, path=None, checkpoint=None):
        assert path is not None or checkpoint is not None, 'missing checkpoint source (path/checkpoint_dict)'
        checkpoint = checkpoint or torch.load(path)
        drmade = DRMADE.load_from_checkpoint(checkpoint=checkpoint['models']['drmade'])
        self.__init__(hparams=checkpoint.get(constants.HPARAMS_DICT, None), drmade=drmade)
        drmade = drmade.to(self.get(constants.DEVICE))
        drmade.encoder = drmade.encoder.to(self.get(constants.DEVICE))
        drmade.decoder = drmade.decoder.to(self.get(constants.DEVICE))
        drmade.made = drmade.made.to(self.get(constants.DEVICE))
        for name, state_dict in checkpoint.get('schedulers', dict()).items():
            self.schedulers_list[self.get(f'{constants.SCHEDULER_PREFIX}{name}/index')].load_state_dict(state_dict)
        for name, state_dict in checkpoint.get('optimizers', dict()).items():
            self.optimizers_list[self.get(f'{constants.OPTIMIZER_PREFIX}{name}/index')].load_state_dict(state_dict)

    def save_model(self, output_path=None):
        output_path = output_path or self.get('check_point_saving_dir')
        self.get('drmade').save(output_path + f'/{self.get("name")}-E{self.get("epoch")}.pth')

    def _evaluate_loop(self, data_loader, record_input_images=False, record_reconstructions=False):
        with torch.no_grad():
            log_prob = torch.Tensor().to(self.get(constants.DEVICE))
            decoder_loss = torch.Tensor().to(self.get(constants.DEVICE))
            reconstructed_images = torch.Tensor().to(self.get(constants.DEVICE))
            features = torch.Tensor().to(self.get(constants.DEVICE))
            labels = np.empty(0, dtype=np.int8)
            input_images = torch.Tensor().to(self.get(constants.DEVICE))
            for batch_idx, (images, label) in enumerate(data_loader):
                images = images.to(self.get(constants.DEVICE))
                if record_input_images:
                    input_images = torch.cat((input_images, images), dim=0)

                output, latent, reconstruction = self.get("drmade")(images)
                decoder_loss = torch.cat(
                    (decoder_loss, self.get("drmade").decoder.distance(images, reconstruction)), dim=0)
                log_prob = torch.cat((log_prob, self.get("drmade").made.log_prob_hitmap(latent).sum(1)), dim=0)
                features = torch.cat((features, latent), dim=0)

                if record_reconstructions:
                    reconstructed_images = torch.cat((reconstructed_images, reconstruction), dim=0)
                labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
        return log_prob, decoder_loss, features, labels, input_images, reconstructed_images

    def _submit_latent(self, features, title=''):
        for i in range(features.shape[1]):
            self.get('writer').add_histogram(f'latent/{title}/{i}', features[:, i], self.get(constants.EPOCH))

    def _submit_extreme_reconstructions(
            self, input_images, reconstructed_images, decoder_loss, title='', num_cases=None):
        num_cases = num_cases or self.get(constants.HPARAMS_DICT).get('num_extreme_cases',
                                                                      model_config.num_extreme_cases)
        distance_hitmap = self.get('drmade').decoder.distance_hitmap(input_images,
                                                                     reconstructed_images).detach().cpu().numpy()

        distance = decoder_loss.detach().cpu().numpy()
        sorted_indexes = np.argsort(distance)
        result_images = np.empty((num_cases * 3, input_images.shape[1], input_images.shape[2], input_images.shape[3]))
        input_images = input_images.cpu()
        reconstructed_images = reconstructed_images.cpu()
        for i, index in enumerate(sorted_indexes[:num_cases]):
            result_images[i * 3] = input_images[index]
            result_images[i * 3 + 1] = reconstructed_images[index]
            result_images[i * 3 + 2] = distance_hitmap[index]
        self.get('writer').add_images(f'best_reconstruction/{title}', result_images, self.get(constants.EPOCH))

        result_images = np.empty((num_cases * 3, input_images.shape[1], input_images.shape[2], input_images.shape[3]))
        for i, index in enumerate(sorted_indexes[-1:-(num_cases + 1):-1]):
            result_images[i * 3] = input_images[index]
            result_images[i * 3 + 1] = reconstructed_images[index]
            result_images[i * 3 + 2] = distance_hitmap[index]

        self.get('writer').add_images(f'worst_reconstruction/{title}', result_images, self.get(constants.EPOCH))

    def evaluate(self, evaluation_interval=1):
        self.setup_writer()  # in case it hasn't already been setup

        self.get('drmade').eval()

        # variables
        epoch = self.get(constants.EPOCH)
        hparams = self.get(constants.HPARAMS_DICT)

        num_extreme_cases = hparams.get('num_extreme_cases', model_config.num_extreme_cases)
        track_extreme_reconstructions = num_extreme_cases and hparams.get('track_extreme_reconstructions',
                                                                          model_config.num_extreme_cases)
        track_jacobian_interval = hparams.get('track_jacobian_interval', model_config.track_jacobian_interval)
        track_jacobian_random_selection = hparams.get('track_jacobian_random_selection', None)
        track_jacobian = track_jacobian_interval and ((epoch // evaluation_interval) % track_jacobian_interval == 0)

        submit_latent_interval = hparams.get('submit_latent_interval', model_config.submit_latent_interval)
        submit_latent = submit_latent_interval and (epoch // evaluation_interval) % submit_latent_interval == 0

        evaluate_train_interval = hparams.get('evaluate_train_interval', model_config.evaluate_train_interval)

        # evaluate test data
        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.get('test_loader'), track_jacobian or track_extreme_reconstructions, track_extreme_reconstructions)

        # auc calculation
        self.get('writer').add_scalar(
            f'auc/decoder',
            roc_auc_score(y_true=np.isin(labels, self.get('normal_classes')).astype(np.int8),
                          y_score=(-decoder_loss).cpu()),
            epoch)
        self.get('writer').add_scalar(
            f'auc/made',
            roc_auc_score(y_true=np.isin(labels, self.get('normal_classes')).astype(np.int8), y_score=log_prob.cpu()),
            epoch)

        anomaly_indexes = (np.isin(labels, hparams.get('normal_classes', self.get('normal_classes'))) == False)

        # loss histograms
        self.get('writer').add_histogram(f'loss/decoder/test/anomaly', decoder_loss[anomaly_indexes], epoch)
        self.get('writer').add_histogram(f'loss/decoder/test/normal', decoder_loss[(anomaly_indexes == False)], epoch)

        self.get('writer').add_histogram(f'loss/made/anomaly', log_prob[anomaly_indexes], epoch)
        self.get('writer').add_histogram(f'loss/made/normal', log_prob[(anomaly_indexes == False)], epoch)

        if track_extreme_reconstructions:
            self._submit_extreme_reconstructions(
                images[anomaly_indexes], reconstruction[anomaly_indexes], decoder_loss[anomaly_indexes],
                'test/anomaly', num_extreme_cases)
            self._submit_extreme_reconstructions(
                images[(anomaly_indexes == False)], reconstruction[(anomaly_indexes == False)],
                decoder_loss[(anomaly_indexes == False)], 'test/normal', num_extreme_cases)

        if submit_latent:
            self._submit_latent(features[anomaly_indexes], 'test/anomaly')
            self._submit_latent(features[(anomaly_indexes == False)], 'test/normal')

        if track_jacobian:
            self._track_jacobian(images[anomaly_indexes], 'test/anomaly', track_jacobian_random_selection)
            self._track_jacobian(images[(anomaly_indexes == False)], 'test/normal', track_jacobian_random_selection)

        if evaluate_train_interval and (epoch // evaluation_interval) % evaluate_train_interval == 0:
            log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
                self.get('train_loader'), track_jacobian or track_extreme_reconstructions,
                track_extreme_reconstructions)

            self.get('writer').add_histogram(f'loss/decoder/train', decoder_loss, epoch)
            self.get('writer').add_histogram(f'loss/made/train', log_prob, epoch)

            if submit_latent:
                self._submit_latent(features, 'train')

            if track_extreme_reconstructions:
                self._submit_extreme_reconstructions(images, reconstruction, decoder_loss, 'train', num_extreme_cases)

            if track_jacobian:
                self._track_jacobian(images, 'train', track_jacobian_random_selection)

        self.get('writer').flush()
        self.get('drmade').train()

    def _track_jacobian(self, inputs, title='', random_selection=None):
        if random_selection:
            permutation = torch.randperm(inputs.size(0), device=self.get(constants.DEVICE))
            inputs = inputs[permutation[:random_selection]]
        input_shape = self.get('train_data').input_shape()
        mean = torch.zeros(self.get('drmade').encoder.latent_size, input_shape[0], input_shape[1], input_shape[2],
                           device=self.get(constants.DEVICE))
        count = inputs.shape[0]
        for i in range(count):
            jac = jacobian(self.get('drmade').encoder,
                           inputs[i].view(-1, input_shape[0], input_shape[1], input_shape[2]))[0, :, 0]
            mean += jac / count
        total_mean = mean.mean(dim=0, keepdim=True)
        self.get('writer').add_images(f'jacobian/latent_input/{title}', mean, self.get(constants.EPOCH))
        self.get('writer').add_images(f'jacobian/latent_input/mean/{title}', total_mean, self.get(constants.EPOCH))
        # self.get('writer').add_scalars(
        #     f'jacobian_norm/latent_input/{title}', {f'{i}': mean[i].norm() for i in range(mean.shape[0])},
        #     self.get(constants.EPOCH))
        self.get('writer').add_scalar(
            f'jacobian_norm/latent_input/mean/{title}', total_mean.norm(), self.get(constants.EPOCH))

    def submit_embedding(self, data_loader='test_loader'):
        self.setup_writer()  # in case it hasn't already been setup
        self.get('drmade').eval()

        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.get(data_loader),
            record_input_images=True,
        )
        self.get('writer').add_embedding(
            features, metadata=labels, label_img=images,
            global_step=self.get(constants.EPOCH),
            tag=self.get(constants.TRAINER_NAME))
        self.get('writer').flush()
        self.get('drmade').train()

    def train(self):
        self.setup_writer()  # in case it hasn't already been setup

        # training hyper parameters
        evaluation_interval = self.get(constants.HPARAMS_DICT).get(
            'evaluation_interval', model_config.evaluation_interval)
        embedding_interval = self.get(constants.HPARAMS_DICT).get(
            'embedding_interval', model_config.embedding_interval)
        save_interval = self.get(constants.HPARAMS_DICT).get('save_interval', model_config.save_interval)
        start_epoch = self.get(constants.HPARAMS_DICT).get('start_epoch', 0)
        max_epoch = self.get(constants.HPARAMS_DICT).get('max_epoch', model_config.max_epoch)

        print('Starting Training - intervals:[',
              'evaluation:{}, embedding:{}, save:{}, start_epoch:{}, max_epoch:{} ]'.format(
                  evaluation_interval, embedding_interval, save_interval, start_epoch, max_epoch))

        for epoch in range(start_epoch, max_epoch):
            self.set(constants.EPOCH, epoch, replace=True)
            print(f'epoch {self.get(constants.EPOCH):5d}')
            if evaluation_interval and epoch % evaluation_interval == 0:
                if self.verbose:
                    print('\t+ evaluating')
                self.evaluate(evaluation_interval)

            if embedding_interval and (epoch + 1) % embedding_interval == 0:
                if self.verbose:
                    print('\t+ submitting embedding')
                self.submit_embedding()

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_model()

            for loop in self.loops_list:
                active = loop.is_active(self.context)
                if self.verbose:
                    print(f'\t+ calling loop {loop.name} - active:{active}')
                if active:
                    self.set(f'{constants.LOOP_PREFIX}{loop.name}/data', loop(self.context), replace=True)
                    if self.verbose:
                        print(f'\t+ submitting loop {loop.name} data')
                    loop.submit_loop_data(self.context)

            for index, scheduler in enumerate(self.schedulers_list):
                if self.verbose:
                    print(f'\t+ scheduler {self.get(f"{constants.SCHEDULER_PREFIX}index/{index}")} step')
                scheduler.step()

            self.submit_progress()
