import os
import logging
import pickle

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from ...utils import load_pickle_obj
from ..flavor_wheel import FlavorWheel
from .mass_spectrum import MassSpectrumDataset
from .nir_spectrum import NIRSpectrumDataset
from .utils import random_initial_seed

#--------------------------------------------------------------------------------
# base.py containes the object which inherited the torch.utils.data.Dataset
# for deep neural network training
# the object is used in ML/DL classifier training
#--------------------------------------------------------------------------------


__all__ = ['GCMSBaseDataset', 'NIRBaseDataset']


class GCMSBaseDataset(Dataset):
    def __init__(self,
            path,
            wheel_path,
            select = 'all',
            datatype = '2d',
            mode = 'train',
            split_ratio = None,
            seed = None,
            val_seed = None,
            flatten = False,
            normalize = False,
            search = False,
            reduce_blank = True):

        super(GCMSBaseDataset, self).__init__()

        self.select_list = ['fragrance', 'aroma', 'all']
        if select not in self.select_list:
            raise ValueError(str(select) + ' is not valid for Dataset: GCMSClassifierDataset.')

        self.mode_list = ['train', 'test', 'val']
        if mode not in self.mode_list:
            raise ValueError('Argument: mode must in ' + str(self.mode_list))

        self.normalize_list = ['whole', 'time', 'm/z', 'none']
        self.normalize_dict = {
                'whole': {'sum_axis': (0, 1, 2), 'retain_axis': None},
                'time': {'sum_axis': (0, 2), 'retain_axis': 1},
                'm/z': {'sum_axis': (0, 1), 'retain_axis': 2},
                }

        if normalize.lower() not in self.normalize_list:
            raise ValueError(normalize, ' is not a valid normalization selection.')

        if not isinstance(flatten, bool):
            raise TypeError('Argument: flatten must be a boolean [True or False].')

        if not isinstance(reduce_blank, bool):
            raise TypeError('Argument: reduce_blank must be a boolean [True or False].')

        self.path = path
        self.wheel_path = wheel_path
        self.select = select
        self.datatype = datatype
        self.mode = mode
        self.flatten = flatten
        self.normalize = normalize
        self.search = search
        self.reduce_blank = reduce_blank
        self.size_change = False

        if seed is None:
            self.seed = random_initial_seed()
        else:
            self.set_seed(seed, val = False)

        if val_seed is None:
            self.val_seed = random_initial_seed()
        else:
            self.set_seed(val_seed, val = True)

        if split_ratio is None:
            self.split_ratio = {'train': 1.0, 'val': 0.0, 'test': 0.0}
        else:
            if split_ratio['train'] + split_ratio['val'] + split_ratio['test'] != 1.:
                raise RuntimeError('Summation of train, val and testing set must be 1.')

            self.split_ratio = split_ratio

        self.flavor_wheel = FlavorWheel(wheel_path)
        mass_spectrum_dataset = MassSpectrumDataset(path,
                datatype = datatype, search = search)

        self.mass_length = mass_spectrum_dataset.mass_length
        self.time_length = mass_spectrum_dataset.time_length

        if select == 'all':
            tmp_aroma = mass_spectrum_dataset.dataset(select = 'aroma',
                    reduce_blank = reduce_blank)
            tmp_fragrance = mass_spectrum_dataset.dataset(select = 'fragrance',
                    reduce_blank = reduce_blank)

            self.mass_spectra = np.concatenate((tmp_fragrance[0], tmp_aroma[0]), axis = 0)
            self.labels = self._concatenate_labels(tmp_fragrance[1], tmp_aroma[1])
            self.data_indices = tmp_fragrance[2] + tmp_aroma[2]
            self.label_descriptions = tmp_fragrance[3] + tmp_aroma[3]
        else:
            tmp = mass_spectrum_dataset.dataset(select = self.select, reduce_blank = reduce_blank)
            self.mass_spectra, self.labels, self.data_indices, self.label_descriptions = tmp

        self._split_data()

    def set_seed(self, seed, val = False):
        # set the random seed of the dataset
        if not isinstance(seed, int):
            raise TypeError('Seed in dataset must be a int.')

        if val:
            self.val_seed = seed
        else:
            self.seed_change = True
            self.seed = seed

        return None

    def resplit(self, seed = None, val = True):
        if seed is None:
            seed = random_initial_seed()

        self.set_seed(seed = seed, val = val)
        self._split_data()

        return None

    def set_mode(self, mode):
        # set mode of the dataset
        if mode not in self.mode_list:
            raise ValueError('Argument: mode must in ' + str(self.mode_list))

        self.mode = mode

        return None

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def _concatenate_labels(self, fragrance, aroma):
        labels = np.zeros((fragrance.shape[0] + aroma.shape[0], fragrance.shape[1] + aroma.shape[1]))
        labels[: fragrance.shape[0], : fragrance.shape[1]] = fragrance
        labels[fragrance.shape[0]:, fragrance.shape[1]: ] = aroma
        return labels

    def _split_data(self):
        # train, val, test set will be build in this function
        if self.split_ratio['test'] <= 0:
            train_val_mass_spectra, train_val_labels, train_val_indices = self.mass_spectra.copy(), self.labels.copy(), self.data_indices.copy()
            test_mass_spectra, test_labels, test_indices = np.array([[[]]]), np.array([[]]), []
        else:
            tmp = train_test_split(self.mass_spectra, self.labels, self.data_indices,
                    test_size = self.split_ratio['test'], random_state = self.seed)
            train_val_mass_spectra, test_mass_spectra, train_val_labels, test_labels, train_val_indices, test_indices = tmp

        if self.normalize != 'none' and (self.seed_change or self.size_change):
            if self.split_ratio['test'] < 1.:
                self._normalize_spectra(train_val_mass_spectra, mode = self.normalize, reset_memory = True)

            # ensure there are numbers in the array
            if sum(test_mass_spectra.shape) >= len(test_mass_spectra.shape):
                test_mass_spectra = self._normalize_spectra(test_mass_spectra, mode = self.normalize)

            self.seed_change, self.size_change = False, False

        if self.flatten:
            test_mass_spectra = self._flatten_spectra(test_mass_spectra)

        test_set = {'mass_spectra': test_mass_spectra, 'labels': test_labels, 'indices': test_indices}
        if self.split_ratio['test'] > 0:
            self._check_dataset_num(test_set)

        split_val_size = self.split_ratio['val'] / (self.split_ratio['train'] + self.split_ratio['val'])
        if split_val_size <= 0.:
            train_mass_spectra, train_labels, train_indices = train_val_mass_spectra, train_val_labels, train_val_indices
            val_mass_spectra, val_labels, val_indices = np.array([[[]]]), np.array([[]]), []
        else:
            tmp = train_test_split(train_val_mass_spectra, train_val_labels, train_val_indices,
                    test_size = split_val_size, random_state = self.val_seed)

            train_mass_spectra, val_mass_spectra, train_labels, val_labels, train_indices, val_indices = tmp

        if self.normalize != 'none':
            # ensure there are numbers in the array
            if sum(train_mass_spectra.shape) > len(train_mass_spectra.shape):
                train_mass_spectra = self._normalize_spectra(train_mass_spectra, mode = self.normalize)

            # ensure there are numbers in the array
            if sum(val_mass_spectra.shape) > len(val_mass_spectra.shape):
                val_mass_spectra = self._normalize_spectra(val_mass_spectra, mode = self.normalize)

        if self.flatten:
            train_mass_spectra = self._flatten_spectra(train_mass_spectra)
            val_mass_spectra = self._flatten_spectra(val_mass_spectra)

        train_set = {'mass_spectra': train_mass_spectra, 'labels': train_labels, 'indices': train_indices}
        if self.split_ratio['train'] > 0.:
            self._check_dataset_num(train_set)

        val_set = {'mass_spectra': val_mass_spectra, 'labels': val_labels, 'indices': val_indices}
        if self.split_ratio['val'] > 0.:
            self._check_dataset_num(val_set)

        self.data = {'train': train_set, 'val': val_set, 'test': test_set}

        return None

    def _flatten_spectra(self, spectra):
        if not isinstance(spectra, np.ndarray):
            raise TypeError('The spectra data must be a numpy.ndarray.')

        sample_num = spectra.shape[0]
        spectra = spectra.reshape(sample_num, -1)

        return spectra

    def _normalize_spectra(self, spectra, mode, reset_memory = False):
        if reset_memory:
            new_shape = [1 for i in range(len(spectra.shape))]
            retain_axis = self.normalize_dict[mode]['retain_axis']
            if retain_axis is not None:
                new_shape[retain_axis] = spectra.shape[retain_axis]
                new_shape = tuple(new_shape)

            self.spectra_mean = spectra.mean(axis = self.normalize_dict[mode]['sum_axis']).reshape(new_shape)
            self.spectra_std = spectra.std(axis = self.normalize_dict[mode]['sum_axis']).reshape(new_shape)

        spectra = (spectra - self.spectra_mean) / self.spectra_std

        return spectra

    def _check_dataset_num(self, dataset):
        assert dataset['mass_spectra'].shape[0] == dataset['labels'].shape[0]
        assert dataset['mass_spectra'].shape[0] == len(dataset['indices'])
        return None


class NIRBaseDataset(Dataset):
    def __init__(self,
            path,
            label_list,
            select = 'flavor',
            mode = 'train',
            file_name = 'flavor_spectrum.xlsx',
            split_ratio = None,
            seed = None,
            val_seed = None,
            use_previous_batch = False):

        super(NIRBaseDataset, self).__init__()

        self.spectrum_dataset = NIRSpectrumDataset(
                path,
                label_list,
                file_name = file_name,
                use_previous_batch = use_previous_batch)

        self.file_name = file_name
        self.path = self.spectrum_dataset.path
        self.label_list = self.spectrum_dataset.label_list
        self.label_descriptions = self.spectrum_dataset.label_descriptions

        self.select_list = ['flavor', 'agtron']
        if select not in self.select_list:
            raise ValueError(str(select) + ' is not valid for Dataset: NIRClassifierDataset.')

        self.mode_list = ['train', 'test', 'val']
        if mode not in self.mode_list:
            raise ValueError('Argument: mode must in ' + str(self.mode_list))

        self.select = select
        self.mode = mode
        self.use_previous_batch = use_previous_batch

        if seed is None:
            self.seed = random_initial_seed()
        else:
            self.set_seed(seed, val = False)

        if val_seed is None:
            self.val_seed = random_initial_seed()
        else:
            self.set_seed(val_seed, val = True)
        
        if split_ratio is None:
            self.split_ratio = {'train': 1.0, 'val': 0.0, 'test': 0.0}
        else:
            if split_ratio['train'] + split_ratio['val'] + split_ratio['test'] != 1.:
                raise RuntimeError('Summation of train, val and testing set must be 1.')

            self.split_ratio = split_ratio

        self._split_data()

    def set_seed(self, seed, val = False):
        # set the random seed of the dataset
        if not isinstance(seed, int):
            raise TypeError('Seed in dataset must be a int.')

        if val:
            self.val_seed = seed
        else:
            self.seed_change = True
            self.seed = seed

        return None

    def resplit(self, seed = None, val = True):
        if seed is None:
            seed = random_initial_seed()

        self.set_seed(seed = seed, val = val)
        self._split_data()

        return None

    def set_mode(self, mode):
        # set mode of the dataset
        if mode not in self.mode_list:
            raise ValueError('Argument: mode must in ' + str(self.mode_list))

        self.mode = mode

        return None

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def _split_data(self):
        select_index = [i for i in range(len(self.spectrum_dataset))]
        min_ratio = 1 / len(self.spectrum_dataset)
        if self.split_ratio['test'] < min_ratio:
            train_val_index, test_index = select_index, []
        elif self.split_ratio['test'] > (1 - min_ratio):
            train_val_index, test_index = [], select_index
        else:
            train_val_index, test_index = train_test_split(select_index,
                    test_size = self.split_ratio['test'], random_state = self.seed)

        test_spectra, test_labels = self.spectrum_dataset.dataset(self.select, index_list = test_index)
        test_set = {'nir_spectra': test_spectra, 'labels': test_labels}
        self._check_dataset_num(test_set)

        split_val_size = self.split_ratio['val'] / (self.split_ratio['train'] + self.split_ratio['val'])
        min_ratio = 1 / len(train_val_index)
        if split_val_size < min_ratio:
            train_index, val_index = train_val_index, []
        elif split_val_size > (1 - min_ratio):
            train_index, val_index = [], train_val_index
        else:
            train_index, val_index = train_test_split(train_val_index, 
                    test_size = split_val_size, random_state = self.val_seed)

        train_spectra, train_labels = self.spectrum_dataset.dataset(self.select, index_list = train_index)
        train_set = {'nir_spectra': train_spectra, 'labels': train_labels}
        self._check_dataset_num(train_set)

        val_spectra, val_labels = self.spectrum_dataset.dataset(self.select, index_list = val_index)
        val_set = {'nir_spectra': val_spectra, 'labels': val_labels}
        self._check_dataset_num(val_set)

        self.sample_number = {'train': len(train_index), 'val': len(val_index), 'test': len(test_index)}
        self.data = {'train': train_set, 'val': val_set, 'test': test_set}

        return None

    def _check_dataset_num(self, dataset):
        assert dataset['nir_spectra'].shape[0] == dataset['labels'].shape[0]
        return None


