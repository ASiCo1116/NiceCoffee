import os
import csv
import logging
import datetime

import numpy as np

from .base import BaseDemoObject
from .utils import init_torch_device, init_directory

from ..model import load_model, ModelExtensionDict
from ..data.dataset import NIRClassifierDataset, GCMSClassifierDataset

#--------------------------------------------------------------------------------
# classifer.py contain the object used for NIR and GCMS classifer
#--------------------------------------------------------------------------------

class ClassifierDemo(BaseDemoObject):
    def __init__(self,
            config_path,
            datatype = 'gcms',
            best_state = True,
            block_length = 25,
            terminal_length = 150):

        super(ClassifierDemo, self).__init__(
                config_path = config_path,
                block_length = block_length,
                terminal_length = terminal_length)

        if not isinstance(datatype, str):
            raise TypeError('Argument: datatype must be a string.')
        
        if datatype.lower() not in ['nir', 'gcms']:
            raise ValueError('Argument: datatype in lib.trainer.MachineLearningTrainer must be NIR or GCMS.')

        self.datatype = datatype

        self.dataset = self._init_dataset(self.config)

        try:
            self.best_state_usage = self.config['best_state']
        except KeyError:
            self.best_state_usage = best_state

        if not isinstance(self.best_state_usage, bool):
            raise TypeError('best_state in config file must be a boolean.')

        self.model_path_list, self.model_name_list = self._search_model_path(self.config, self.best_state_usage)

        try:
            device_index = self.config['device']
        except KeyError:
            device_index = None

        self.device = init_torch_device(device_index)

        self.models = []
        self.output_size = None
        for path in self.model_path_list:
            tmp_model, tmp_device = load_model(path, device = self.device)
            self.models.append(tmp_model)
            if self.output_size is None:
                self.output_size = tmp_model.output_size
            else:
                assert self.output_size == tmp_model.output_size

        logging.info('Model and dataset init done, start predict data.')

    def _init_dataset(self, config = None):
        if config is None:
            config = self.config

        if self.datatype.lower() == 'nir':
            dataset = NIRClassifierDataset(**config['dataset'])
        elif self.datatype.lower() == 'gcms':
            dataset = GCMSClassifierDataset(**config['dataset'])

        return dataset

    def _search_model_path(self, config, best_state):
        try:
            dir_path = config['path']
        except KeyError:
            logging.warning('path is not set in config, use default directory: outputs.')
            dir_path = 'outputs'

        try:
            model_list = config['model']
        except KeyError:
            raise RuntimeError('model is recommended to be set as a list object in yaml config.')

        if not isinstance(model_list, (list, str)):
            raise TypeError('model is recommended to be set as a list object in yaml config.')

        if isinstance(model_list, str):
            model_list = [model_list]

        model_path_list, model_name_list = [], []
        extensions = list(ModelExtensionDict.keys())
        for model_name in model_list:
            if not isinstance(model_name, str):
                raise TypeError('model name set in the model list in the config must be a string.')

            tmp_model_count = 0
            tmp_dir = os.path.join(dir_path, model_name)
            for f in os.listdir(tmp_dir):
                if best_state:
                    if 'best_state' in f:
                        model_file = f
                        break

                for extension in extensions:
                    if f.endswith(extension):
                        model_file = f
                        tmp_model_count += 1

            tmp_model_path = os.path.join(dir_path, model_name, model_file)
            model_name_list.append(model_name)
            model_path_list.append(tmp_model_path)

            if tmp_model_count > 1:
                logging.warning('Find more than one model in the path: ' + str(tmp_dir))
                logging.warning('Note that the model used in this directory is model: ' + str(model_file) + '\n')

        return model_path_list, model_name_list

    def _demo(self, config):
        if config is None:
            config = self.config

        if self.datatype == 'gcms':
            try:
                sample_odor_type = config['sample_odor_type']
            except KeyError:
                logging.info('GCMS demo must set the dict:sample_odor_type to grab mass spectrum of the selected sample.')
                exit()

        descriptions = self.dataset.descriptions()
        assert len(descriptions) == self.output_size

        sample_name = config['sample_name']
        if not isinstance(sample_name, list):
            raise TypeError('Argument: sample_name in config file must be a list.')

        spectra, ground_truth, samples = [], [], []
        for name in sample_name:
            if not isinstance(name, str):
                raise TypeError('Sample name in the list:sample_name must be the str:name of the sample.')

            if self.datatype == 'gcms':
                if isinstance(sample_odor_type[name], list):
                    for i in range(len(sample_odor_type[name])):
                        temp_name = self._gcms_name_transform(name, sample_odor_type[name][i])
                        spectrum, label = self.dataset.catch_sample_by_name(temp_name)
                        spectra.append(spectrum)
                        ground_truth.append(label)
                        samples.append(temp_name)

                elif isinstance(sample_odor_type[name], str):
                    temp_name = self._gcms_name_transform(name, sample_odor_type[name])
                    spectrum, label = self.dataset.catch_sample_by_name(temp_name)
                    spectra.append(spectrum)
                    ground_truth.append(label)
                    samples.append(temp_name)

            elif self.datatype == 'nir':
                spectrum, label = self.dataset.catch_sample_by_name(name, mode = 'mean')
                spectra.append(spectrum)
                ground_truth.append(label[:, : -1])
                samples.append(name)

        ground_truth = np.concatenate(ground_truth, axis = 0)

        spectra = self._pack_spectra(spectra)
        predictions, shape = [], None
        for model_index in range(len(self.models)):
            result = self.models[model_index].predict(spectra, return_type = 'numpy')
            if shape is None:
                shape = result.shape
            else:
                if result.shape != shape:
                    raise RuntimeError('The size of output of the selected model is different, this is not valid in demo API.')

            predictions.append(result)

        # prediction dim: (model, sample, 1d-label)
        predictions = np.stack(predictions, axis = 0)
        lines, csv_lines = self._display_result(predictions, ground_truth, descriptions, samples, self.model_name_list)

        return (lines, csv_lines)

    def _display_result(self, predictions, ground_truth, descriptions, sample_name_list, model_name):
        lines, csv_lines = [], []
        for sample_index in range(len(sample_name_list)):
            temp_models = model_name.copy()
            temp_models.append('GroundTruth')
            sample_result = predictions[:, sample_index, :]
            sample_result = np.concatenate((sample_result, np.expand_dims(ground_truth[sample_index], axis = 0)), axis = 0)

            logging.info('=' * self.terminal_length)
            lines.append('=' * self.terminal_length + '\n')

            logging.info('Sample name: ' + sample_name_list[sample_index])
            lines.append('Sample name: ' + sample_name_list[sample_index] + '\n')
            csv_lines.append('Sample name: ' + sample_name_list[sample_index] + '\n')

            logging.info('-' * self.terminal_length)
            lines.append('-' * self.terminal_length + '\n')

            title_line = 'Model'.rjust(self.block_length)
            csv_title_line = 'Model,'
            model_lines = [temp_models[i].rjust(self.block_length) for i in range(len(temp_models))]
            csv_model_lines = [(temp_models[i] + ',') for i in range(len(temp_models))]
            for category_index in range(len(descriptions)):
                title_line += descriptions[category_index].rjust(self.block_length)
                csv_title_line += descriptions[category_index] + ','
                for model_index in range(len(temp_models)):
                    result = self._decode_result(sample_result[model_index, category_index])
                    model_lines[model_index] += result.rjust(self.block_length)
                    csv_model_lines[model_index] += (result + ',')

                if (category_index + 1) % (self.block_num - 1) == 0:
                    logging.info(title_line)
                    lines.append(title_line + '\n')
                    for category_result in model_lines:
                        logging.info(category_result)
                        lines.append(category_result + '\n')

                    title_line = 'Model'.rjust(self.block_length)
                    model_lines = [temp_models[i].rjust(self.block_length) for i in range(len(temp_models))]

                    logging.info('-' * self.terminal_length)
                    lines.append('-' * self.terminal_length + '\n')

                elif category_index == len(descriptions) - 1:
                    logging.info(title_line)
                    lines.append(title_line + '\n')
                    csv_lines.append(csv_title_line + '\n')
                    for category_result in model_lines:
                        logging.info(category_result)
                        lines.append(category_result + '\n')

                    for category_result in csv_model_lines:
                        csv_lines.append(category_result + '\n')
                    
            logging.info('=' * self.terminal_length + '\n')
            lines.append('=' * self.terminal_length + '\n\n')

        return lines, csv_lines

    def _decode_result(self, result, threshold = 0.5):
        return 'yes' if result > threshold else 'no'

    def _gcms_name_transform(self, name, odor_type):
        if odor_type.lower() not in ['fragrance', 'aroma']:
            raise ValueError('Odor type in dict:sample_odor_type must be str:fragrance or str:aroma.')

        if odor_type.lower() == 'fragrance':
            name += '_Dry'
        elif odor_type.lower() == 'aroma':
            name += '_Wet'

        return name

    def _pack_spectra(self, spectra_list):
        least_size, sample_num = None, 0
        for spectrum in spectra_list:
            if least_size is None:
                least_size = list(spectrum.shape[1: ])
            else:
                temp_shape = spectrum.shape
                assert len(temp_shape[1: ]) == len(least_size)
                for dim in range(len(least_size)):
                    if temp_shape[dim + 1] < least_size[dim]:
                        least_size[dim + 1] = temp_shape[dim]

            sample_num += 1                

        array_size = least_size.copy()
        array_size.insert(0, sample_num)
        spectra_array = np.empty(tuple(array_size))
        slc = [slice(None)] * len(least_size)
        for index in range(len(slc)):
            slc[index] = slice(0, least_size[index])

        for index in range(len(spectra_list)):
            temp_slc = slc.copy()
            temp_slc.insert(0, index)
            spectra_array[tuple(temp_slc)] = spectra_list[index]

        return spectra_array

    def _output_file(self, result, config = None):
        if config is None:
            config = self.config

        lines, csv_lines = result
        filename = config['filename']
        save_dir = config['save_dir']
        init_directory(save_dir)

        if config['output_csv_file'] or config['output_txt_file']:
            logging.info('Start output files ...')

        if config['output_csv_file']:
            path = os.path.join(save_dir, filename + '.csv')
            if os.path.isfile(path):
                if not config['overwrite']:
                    path =  os.path.join(save_dir,
                            filename + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f") + '.csv')

            f = open(path, 'w')
            f.writelines(csv_lines)
            f.close()

        if config['output_txt_file']:
            path = os.path.join(save_dir, filename + '.txt')
            if os.path.isfile(path):
                if not config['overwrite']:
                    path =  os.path.join(save_dir,
                            filename + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f") + '.txt')

            f = open(path, 'w')
            f.writelines(lines)
            f.close()

        if config['output_csv_file'] or config['output_txt_file']:
            logging.info('Output files success !!!')

        return None


