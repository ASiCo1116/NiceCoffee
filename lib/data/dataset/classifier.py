import math
import random
import logging

import numpy as np

import sklearn
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..transforms import Compose, ToTensor, NIRPreprocess, select_transforms, combine_transforms_and_preprocess
from ..transforms.gcms_preprocess import GCMSPreprocess, GCMSDatatype, GCMSReduceBlankSample, GCMSFlatten, GCMSReduceDimension 

from .utils import transform_labels 
from .base import GCMSBaseDataset, NIRBaseDataset

#--------------------------------------------------------------------------------
# classifer.py contains the object which inherited from lib.data.dataset.base.BaseDataset
# the object was used in DL classifer training.
#--------------------------------------------------------------------------------


__all__ = ['GCMSClassifierDataset', 'NIRClassifierDataset']


class GCMSClassifierDataset(GCMSBaseDataset):
    def __init__(self,
            path,
            wheel_path,
            select = 'all',
            datatype = '2d',
            mode = 'train',
            wheel_description_transform = 'no',
            normalize = 'none',
            block_length = 15,
            terminal_length = 120,
            transforms = None,
            split_ratio = None,
            seed = None,
            val_seed = None,
            reduce_dim = None,
            flatten = False,
            search = False,
            unit_vector = True,
            reduce_blank = True):

        super(GCMSClassifierDataset, self).__init__(
                path = path,
                wheel_path = wheel_path,
                select = select,
                datatype = datatype,
                mode = mode,
                split_ratio = split_ratio,
                seed = seed,
                val_seed = val_seed,
                flatten = flatten,
                normalize = normalize,
                search = search,
                reduce_blank = reduce_blank)

        if not isinstance(block_length, int):
            raise TypeError('Argument: block_length must be a int.')

        if not isinstance(terminal_length, int):
            raise TypeError('Argument: terminal_length must be a int.')

        if not isinstance(unit_vector, bool):
            raise TypeError('Argument: unit_vector must be a bool.')

        self.reduce_dim = reduce_dim
        self.eps = 1e-8
        self.block_length = block_length
        self.terminal_length = terminal_length
        self.block_num = self.terminal_length // self.block_length
        self.unit_vector = unit_vector

        if transforms is None:
            self.transforms = Compose([ToTensor()])
        elif isinstance(transforms, (list, tuple)):
            if len(transforms) <= 0:
                 self.transforms = Compose([ToTensor()])
            else:
                 self.transforms = select_transforms(transforms)
        else:
            raise TypeError('Transforms in the dataset must be a str tuple or str list.')

        # GCMS preprocess API
        self.preprocess = [GCMSDatatype(datatype = datatype)]
        if reduce_blank:
            pass

        if flatten and reduce_dim is None:
            self.preprocess.append(GCMSFlatten())

        self.wheel_description_transform = wheel_description_transform
        if wheel_description_transform.lower() != 'no':
            self.labels, self.label_descriptions = transform_labels(wheel_description_transform,
                    self.flavor_wheel, self.labels, self.label_descriptions)

        if reduce_dim is not None:
            if not isinstance(reduce_dim, dict):
                raise TypeError('Argument: reduce_dim must be a dict.')

            if reduce_dim['type'] not in ['whole', 'mass_spectrum']:
                raise ValueError("reduce_dim['type'] must be whole or mass_spectrum.")

            if reduce_dim['type'] == 'whole':
                if not flatten:
                    logging.warning("If reduce_dim ('whole' mode), flatten will autometically set to True.")
                    self.flatten = True
                    self.mass_spectra = self._flatten_spectra(self.mass_spectra)

                self.mass_spectra, pca = self._reduce_dim_by_PCA(self.mass_spectra, **reduce_dim)
                temp = GCMSReduceDimension(type = 'whole', unit_vector = unit_vector, **reduce_dim)
                temp.add_pca(pca)
                self.preprocess.append(temp)

            elif reduce_dim['type'] == 'mass_spectrum':
                self.mass_spectra, pca = self._reduce_spectrum_dim_by_PCA(self.mass_spectra, **reduce_dim)
                temp = GCMSReduceDimension(unit_vector = unit_vector, **reduce_dim)
                temp.add_pca(pca)
                self.preprocess.append(temp)

            self.size_change = True
            self.spectrum_size = self.mass_spectra.shape[1: ]

        self.preprocess.append(GCMSFlatten())
        self.preprocess = GCMSPreprocess(self.preprocess)
        self._split_data()

    def catch_all(self):
       mass_spectra = self.data[self.mode]['mass_spectra']
       labels = self.data[self.mode]['labels']

       return mass_spectra, labels

    def catch_sample_by_index(self, index, batch = True):
        mass_spectrum = self.data[self.mode]['mass_spectra'][index]
        label = self.data[self.mode]['labels'][index]

        if batch:
            mass_spectrum = np.expand_dims(mass_spectrum, axis = 0)
            label = np.expand_dims(label, axis = 0)

        return mass_spectrum, label

    def catch_sample_by_name(self, name, batch = True):
        index, data_index = 0, None
        for sample_name in self.data_indices:
            if name.lower() == sample_name.lower():
                data_index = index
                break

            index += 1 

        if data_index is None:
            raise RuntimeError('Sample not found in the dataset.')

        mass_spectrum = self.data[self.mode]['mass_spectra'][data_index]
        label = self.data[self.mode]['labels'][data_index]

        if batch:
            mass_spectrum = np.expand_dims(mass_spectrum, axis = 0)
            label = np.expand_dims(label, axis = 0)

        return mass_spectrum, label

    def catch_sample_name(self, index):
        return self.data['indices'][index]

    def catch_all_sample_name(self):
        return self.data['indices']

    def catch_transforms(self, mode = 'both'):
        if mode not in ['preprocess', 'transforms', 'both']:
            raise TypeError('Mode must be preprocess, transforms or both')

        if mode == 'preprocess':
            return self.preprocess
        elif mode == 'transforms':
            return self.transforms
        elif mode == 'both':
            return combine_transforms_and_preprocess(preprocess = self.preprocess,
                    transforms = self.transforms)

    def data_size(self):
        return tuple(self.data[self.mode]['mass_spectra'].shape[1: ])

    def label_size(self):
        return tuple(self.data[self.mode]['labels'].shape[1: ])

    def descriptions(self):
        return tuple(self.label_descriptions)

    def __len__(self):
        return len(self.data[self.mode]['indices'])

    def __getitem__(self, index):
        mass_spectrum = self.data[self.mode]['mass_spectra'][index]
        mass_spectrum = self.transforms(mass_spectrum)

        label = self.data[self.mode]['labels'][index]
        label = torch.tensor(label)

        return mass_spectrum, label

    def label_distribution(self, mode = None):
        if mode is None:
            mode = self.mode

        if mode not in ['train', 'val', 'test']:
            raise ValueError("Argument: mode in ClassifierDataset.label_distribution() must be 'train', 'val' or 'test'.")

        return self.data[mode]['labels'], self.descriptions()

    def summary_label_distribution(self):
        logging.info('\nTraining set distributions (seed: %d, val_seed: %d):' % (self.seed, self.val_seed))
        logging.info('Number:')
        self._summary_subset('train', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('train', ratio = True)

        logging.info('\nValidation set distributions (seed: %d, val_seed: %d):' % (self.seed, self.val_seed))
        logging.info('Number:')
        self._summary_subset('val', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('val', ratio = True)

        logging.info('\nTesting set distributions (seed: %d):' % self.seed)
        logging.info('Number:')
        self._summary_subset('test', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('test', ratio = True)

        logging.info('Finish summary process.')

        return None

    def _summary_subset(self, mode, ratio = False):
        labels, descriptions = self.label_distribution(mode)
        assert labels.shape[1] == len(descriptions)

        if not ratio:
            logging.info('Spectrum Numbers: {0}'.format(labels.shape[0]))

        lines = ['', '']
        for i in range(len(descriptions)):
            lines[0] += descriptions[i].rjust(self.block_length)
            if ratio:
                lines[1] += ('%.4f' % float(labels.sum(axis = 0)[i] / labels.shape[0])).rjust(self.block_length)
            else:
                lines[1] += ('%d' % int(labels.sum(axis = 0)[i])).rjust(self.block_length)

            if i != 0 and i % (self.block_num - 1) == 0:
                logging.info(lines[0])
                logging.info(lines[1])
                lines = ['', '']
            if i == len(descriptions) - 1:
                logging.info(lines[0])
                logging.info(lines[1])

        return None

    def _reduce_dim_by_PCA(self,
            spectra,
            dim = None,
            type = None,
            dilated_number = None,
            mean_number = None,
            stride = None,
            retain_intensity = 'none'):

        if not isinstance(dim, int):
            raise ValueError('Argument: dim must be set in the reduce_dim dict and be a int.')

        warning_flag = False
        if type is not None:
            warning_flag = True
        if dilated_number is not None:
            warning_flag = True
        if mean_number is not None:
            warning_flag = True
        if stride is not None:
            warning_flag = True
        if retain_intensity != 'none':
            warning_flag = True

        if warning_flag:
            logging.warning('Only argument: dim is valid in the whole mode of reduce_dim.')

        if self.unit_vector:
            spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

        model = PCA(n_components = dim, whiten = True)
        model = model.fit(spectra)
        spectra = model.transform(spectra)

        return spectra, model

    def _reduce_spectrum_dim_by_PCA(self,
            spectra,
            dim = None,
            type = '',
            dilated_number = None,
            mean_number = None,
            stride = None,
            retain_intensity = 'none'):

        sample_number = spectra.shape[0]
        if retain_intensity == 'concat' or retain_intensity == 'multiply':
            intensity = np.sum(spectra, axis = 2)
            intensity = self._reduce_feature_by_time(intensity, dilated_number, mean_number, stride)

            if self.unit_vector:
                intensity = intensity / (np.linalg.norm(intensity, axis = 1, keepdims = True) + self.eps)

        spectra = spectra.reshape(-1, spectra.shape[-1])
        if self.unit_vector:
            spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

        # default dim is original data size
        if dim is None:
            dim = self.mass_length

        if dim < self.mass_length:
            model = PCA(n_components = dim, whiten = True)
            model = model.fit(spectra)
            spectra = model.transform(spectra)
        elif dim > self.mass_length or dim <= 0:
            raise ValueError('Argument: dim in reduce_dim must be smaller than origin mass_length: ', self.mass_length, ' and bigger than zero.')

        spectra = spectra.reshape(sample_number, -1, dim)
        spectra = self._reduce_feature_by_time(spectra, dilated_number, mean_number, stride)
        if retain_intensity == 'concat':
            intensity = np.expand_dims(intensity, axis = 2)
            spectra = np.concatenate((spectra, intensity), axis = 2)
        elif retain_intensity == 'multiply':
            spectra *= np.expand_dims(intensity, axis = 2)

        return spectra, model

    def _reduce_feature_by_time(self, spectra, dilated_number, mean_number, stride):
        if dilated_number is not None and mean_number is None:
            spectra = spectra[:, :: dilated_number] 
        elif dilated_number is None and mean_number is not None:
            spectra = self._mean_spectrum_by_time(spectra, mean_number, stride)
        elif dilated_number is not None and mean_number is not None:
            raise RuntimeError('Argument: dilated_number and mean_number can not be used simultaneously.')

        return spectra

    def _mean_spectrum_by_time(self, spectra, mean_number, stride):
        if stride is None:
            stride = mean_number

        spectra = torch.tensor(spectra).unsqueeze(dim = 1)
        numbers = torch.ones(spectra.size(), dtype = torch.double, device = spectra.device)
        layer = nn.Conv2d(1, 1, kernel_size = (mean_number, 1),
                bias = False,
                stride = (stride, 1),
                padding = (mean_number - 1, 0),
                padding_mode = 'same')

        layer.weight = torch.nn.Parameter(
                torch.ones(layer.weight.size(), dtype = torch.double, device = layer.weight.device))

        spectra = layer(spectra)
        numbers = layer(numbers)

        spectra = (spectra / numbers).detach().cpu()
        spectra = np.array(spectra.squeeze(dim = 1))

        return spectra


class NIRClassifierDataset(NIRBaseDataset):
    def __init__(self, 
            path,
            label_list,
            select = 'flavor',
            mode = 'train',
            file_name = 'flavor_spectrum.xlsx',
            block_length = 15,
            terminal_length = 120,
            preprocess = None,
            transforms = None,
            split_ratio = None,
            seed = None,
            val_seed = None,
            use_previous_batch = False):

        super(NIRClassifierDataset, self).__init__(
                path = path,
                label_list = label_list,
                select = select,
                mode = mode,
                file_name = file_name,
                split_ratio = split_ratio,
                seed = seed,
                val_seed = val_seed,
                use_previous_batch = use_previous_batch)


        if not isinstance(block_length, int):
            raise TypeError('Argument: block_length must be a int.')

        if not isinstance(terminal_length, int):
            raise TypeError('Argument: terminal_length must be a int.')

        self.eps = 1e-10
        self.msc_ref = None
        self.block_length = block_length
        self.terminal_length = terminal_length
        self.block_num = self.terminal_length // self.block_length

        if preprocess is not None:
            if not isinstance(preprocess, (list, tuple)):
                raise TypeError('Argument: preprocess must be a list or tuple')

            self.preprocess = NIRPreprocess(preprocess)
        else:
            self.preprocess = preprocess

        if transforms is None:
            self.transforms = Compose([ToTensor()])
        elif isinstance(transforms, (list, tuple)):
            if len(transforms) <= 0:
                 self.transforms = Compose([ToTensor()])
            else:
                 self.transforms = select_transforms(transforms)
        else:
            raise TypeError('Transforms in the dataset must be a str tuple or str list.')

        if self.preprocess is not None:
            self.data = self._preprocess(self.data)

    def catch_all(self):
       nir_spectra = self.data[self.mode]['nir_spectra']
       labels = self.data[self.mode]['labels']

       return nir_spectra, labels

    def catch_sample_by_index(self, index, batch = True):
        nir_spectrum = self.data[self.mode]['nir_spectra'][index]
        label = self.data[self.mode]['labels'][index]

        if not batch:
            nir_spectrum = np.squeeze(nir_spectrum, axis = 0)
            label = np.squeeze(label, axis = 0)

        return nir_spectrum, label

    def catch_sample_by_name(self, name, mode = 'mean', index = None, batch = True):
        try:
            index = self.spectrum_dataset.data_index[name]
        except KeyError:
            raise RuntimeError('Sample:' + str(name) + ' is not in the dataset.')

        nir_spectrum = self.spectrum_dataset.sample_spectrum(name, mode = mode, index = index)
        label = self.spectrum_dataset.data_dict[index][self.select]
        if mode == 'mean':
            label = np.expand_dims(np.mean(label, axis = 0), axis = 0)
        elif mode == 'index':
            label =  np.exapnd_dims(label[index], axis = 0)

        if len(nir_spectrum.shape) == 2:
            if not batch:
                nir_spectrum = np.squeeze(nir_spectrum)
        elif len(nir_spectrum.shape) == 1:
            if batch:
                nir_spectrum = np.expand_dims(nir_spectrum, axis = 0)

        if len(label.shape) == 2:
            if not batch:
                label = np.squeeze(label, axis = 0)
        elif len(label.shape) == 1:
            if batch:
                label = np.expand_dims(label, axis = 0)

        return nir_spectrum, label

    def catch_sample_name(self, index):
        logging.info('The index used in this function is not same as the index in the outputed spectrum and label array.')
        return self.spectrum_dataset.data_dict[index]['name']

    def catch_all_sample_name(self):
        return list(self.spectrum_dataset.data_index.keys())

    def catch_transforms(self, mode = 'both'):
        if mode not in ['preprocess', 'transforms', 'both']:
            raise TypeError('Mode must be preprocess, transforms or both')

        if mode == 'preprocess':
            return self.preprocess
        elif mode == 'transforms':
            return self.transforms
        elif mode == 'both':
            return combine_transforms_and_preprocess(preprocess = self.preprocess,
                    transforms = self.transforms) 

    def data_size(self):
        return tuple(self.data[self.mode]['nir_spectra'].shape[1: ])

    def label_size(self):
        return tuple(self.data[self.mode]['labels'].shape[1: ])

    def descriptions(self):
        return tuple(self.label_descriptions)

    def label_distribution(self, mode = None):
        if mode is None:
            mode = self.mode

        if mode not in ['train', 'val', 'test']:
            raise ValueError("Argument: mode in ClassifierDataset.label_distribution() must be 'train', 'val' or 'test'.")

        if self.select == 'flavor':
            return self.data[mode]['labels'], self.descriptions()
        elif self.select == 'agtron':
            return self.data[mode]['labels'], ['Agtron']

    def __len__(self):
        return self.data[self.mode]['nir_spectra'].shape[0]

    def __getitem__(self, index):
        nir_spectrum = self.data[self.mode]['nir_spectra'][index]
        nir_spectrum = self.transforms(nir_spectrum)
        nir_spectrum = nir_spectrum.unsqueeze(dim = 0) 

        label = self.data[self.mode]['labels'][index]
        label = torch.tensor(label)

        return nir_spectrum, label

    def summary_label_distribution(self):
        logging.info('\nTraining set distributions (seed: %d, val_seed: %d):' % (self.seed, self.val_seed))
        logging.info('Number:')
        self._summary_subset('train', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('train', ratio = True)

        logging.info('\nValidation set distributions (seed: %d, val_seed: %d):' % (self.seed, self.val_seed))
        logging.info('Number:')
        self._summary_subset('val', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('val', ratio = True)

        logging.info('\nTesting set distributions (seed: %d):' % self.seed)
        logging.info('Number:')
        self._summary_subset('test', ratio = False)
        logging.info('Ratio:')
        self._summary_subset('test', ratio = True)

        logging.info('Finish summary process.')

        return None

    def _summary_subset(self, mode, ratio = False):
        labels, descriptions = self.label_distribution(mode)
        assert labels.shape[1] == len(descriptions)

        if not ratio:
            logging.info('Sample numbers: {0}'.format(self.sample_number[mode]))
            logging.info('Spectrum numbers: {0}'.format(labels.shape[0]))

        lines = ['', '']
        for i in range(len(descriptions)):
            lines[0] += descriptions[i].rjust(self.block_length)
            if ratio:
                lines[1] += ('%.4f' % float(labels.sum(axis = 0)[i] / labels.shape[0])).rjust(self.block_length)
            else:
                lines[1] += ('%d' % int(labels.sum(axis = 0)[i])).rjust(self.block_length)

            if i != 0 and i % (self.block_num - 1) == 0:
                logging.info(lines[0])
                logging.info(lines[1])
                lines = ['', '']
            if i == len(descriptions) - 1:
                logging.info(lines[0])
                logging.info(lines[1])

        return None

    def resplit(self, seed = None, val = True):
        super(NIRClassifierDataset, self).resplit(seed = seed, val = val)

        if self.preprocess is not None:
            self.data = self._preprocess(self.data)

        return None

    def _preprocess(self, data):
        data['train']['nir_spectra'] = self.preprocess(data['train']['nir_spectra'], train = True)
        data['val']['nir_spectra'] = self.preprocess(data['val']['nir_spectra'], train = False)
        data['test']['nir_spectra'] = self.preprocess(data['test']['nir_spectra'], train = False)

        return data


