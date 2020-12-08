import os
import time
import math
import logging
import datetime

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch

from .utils import load_config

from ..model import BinaryCrossEntropyLoss, load_model, is_nir_model
from ..data import NIRSpectrumDataset
from ..utils import init_torch_device, init_directory

#--------------------------------------------------------------------------------
# focusing_plot.py contained the focusing plot method used in NIR spectrum flavor
# prediction
#--------------------------------------------------------------------------------

class FocusingPlot(object):
    def __init__(self, config_path):
        # init I/O
        if not isinstance(config_path, str):
            raise TypeError('Argument: config_path must be a string.')

        self.config_path = config_path
        self.config = load_config(config_path)

        self.model_extension = '.dlcls'
        self.limit_size = 32
        self.x_axis = np.arange(700, 2500, 2)
        self.band_width = self.x_axis.shape[0]
        self.color = {'spectrum': 'blue', 'model_focusing': 'red'}
        self.fig_size = {'length': 20, 'height': 14}
        self.axis_name = {'x': 'Wavelength', 'y1': 'Absorbance', 'y2': 'intensity'}

        try:
            self.best_state_usage = self.config['best_state']
        except KeyError:
            self.best_state_usage = best_state

        if not isinstance(self.best_state_usage, bool):
            raise TypeError('best_state in config file must be a boolean.')

        self.model_dir, model_path = self._search_model_path(self.config, self.best_state_usage)
        self.model, self.device = load_model(model_path)
        if not is_nir_model(self.model):
            raise RuntimeError('The model is not a NIRDeepLearningModule, focusing_plot only support NIRDeepLearningModule.')

        init_directory(os.path.join(self.model_dir, 'focusing_plot'))
        
        samples = self.config['sample_name']
        if not isinstance(samples, (list, str)):
            raise TypeError('sample_name in config must be a list or a string of the sample name.')

        if isinstance(samples, str):
            self.samples = [samples]
        else:
            self.samples = samples

        dataset_config = self.config['dataset']
        dataset_config['label_list'] = 'inner'
        self.dataset = NIRSpectrumDataset(**dataset_config)
        self.spectra, self.model_input = self._grab_data_from_dataset(self.dataset, self.samples)

        self.model_focusing = self._calculate_model_focsuing(self.model, self.model_input)

    def plot(self, config = None):
        if config is None:
            config = self.config

        logging.info('Start training process.')
        start_time = time.time()

        assert self.spectra.shape[0] == self.model_focusing.shape[0]
        for i in range(self.spectra.shape[0]):
            self.single_plot(self.spectra[i], self.model_focusing[i], self.samples[i])

        total_time = time.time() - start_time
        logging.info('All plot output done, program finish. Total cost time: ' + str(datetime.timedelta(seconds = total_time)))

        return None

    def single_plot(self, spectrum, model_focusing, sample_name, model_name = None, save = True):
        if not isinstance(spectrum, (np.ndarray, torch.Tensor)):
            raise TypeError('Input NIR spectrum must be a numpy.ndarray or torch.Tensor.')

        if isinstance(spectrum, torch.Tensor):
            spectrum = spectrum.numpy()

        if not isinstance(model_focusing, (np.ndarray, torch.Tensor)):
            raise TypeError('Input model focusing must be a numpy.ndarray or torch.Tensor.')

        if isinstance(model_focusing, torch.Tensor):
            model_focusing = model_focusing.numpy()

        assert spectrum.shape == model_focusing.shape

        plt.figure(figsize = (self.fig_size['length'], self.fig_size['height']))

        fig, ax1 = plt.subplots()

        if model_name is None:
            model_name = self.config['model']

        ax1.set_title('Focusing_plot of model' + model_name)

        ax1.plot(self.x_axis, spectrum, color = self.color['spectrum'])
        ax1.set_xlabel(self.axis_name['x'])
        ax1.set_ylabel(self.axis_name['y1'])

        ax2 = ax1.twinx()
        ax2.bar(self.x_axis, model_focusing, width = 2, color = self.color['model_focusing'])
        ax2.set_ylabel(self.axis_name['y2'])

        fig.tight_layout()

        if save:
            plt.savefig(os.path.join(self.model_dir, 'focusing_plot', sample_name + '.png'))
            logging.info('Picture: ' + sample_name + '.png saving done.')
        else:
            plt.show()

        return None

    def _search_model_path(self, config, best_state):
        try:
            dir_path = config['path']
        except KeyError:
            logging.warning('path is not set in config, use default directory: outputs.')
            dir_path = 'outputs'

        try:
            model = config['model']
        except KeyError:
            raise RuntimeError('You do not wrtie the tested model in yaml config.')

        model_count = 0
        model_dir = os.path.join(dir_path, model)
        for f in os.listdir(model_dir):
            if best_state:
                if 'best_state' in f:
                    model_file = f
                    break

            if f.endswith(self.model_extension):
                model_file = f
                model_count += 1

        if model_count <= 0:
            raise OSError('Model directory do not has model file, please check the directory.')

        model_path = os.path.join(model_dir, model_file)
        if model_count > 1:
            logging.warning('Find more than one model state in the model folder, note that the output file was using model:' + model_path)

        return model_dir, model_path

    def _grab_data_from_dataset(self, dataset, samples):
        spectra = []
        for sample in samples:
            spectra.append(np.expand_dims(dataset.sample_spectrum(sample), axis = 0))

        spectra = np.concatenate(spectra, axis = 0)
        model_input = torch.tensor(spectra)
        model_input = model_input.unsqueeze(dim = 1).float()
        model_input.requires_grad = True

        return spectra, model_input

    def _calculate_model_focsuing(self, model, spectra):
        model_focusing = []

        criterion = BinaryCrossEntropyLoss()
        criterion.to(self.device)

        model = model.eval()
        model = model.to(self.device)

        max_value = 0.
        iter_round = math.ceil(spectra.size(0) / self.limit_size)
        for index in range(iter_round):
            mini_batch_spectra = spectra[index * self.limit_size: (index + 1) * self.limit_size]
            mini_batch_spectra.to(self.device)

            output = model(mini_batch_spectra)
            binaralized_output = (torch.sigmoid(output) > 0.5).float().detach()

            loss = criterion(output, binaralized_output)

            focusing = torch.autograd.grad(loss, mini_batch_spectra)
            focusing = torch.abs(focusing[0].squeeze(dim = 1)).cpu().detach().numpy()

            if np.max(focusing) > max_value:
                max_value = np.max(focusing)

            model_focusing.append(focusing)

        model_focusing = np.concatenate(model_focusing, axis = 0)
        if max_value > 0.:
            model_focusing = model_focusing / max_value

        return model_focusing


