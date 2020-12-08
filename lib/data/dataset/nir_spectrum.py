import os
import logging
import numpy as np

from ..labeler import NIRInfoGrabber

#--------------------------------------------------------------------------------
# nir_spectrum.py contains the object which can acquire NIR spectrum from formated
# excel file.
#--------------------------------------------------------------------------------


__all__ = ['NIRSpectrumDataset']


class NIRSpectrumDataset(object):
    def __init__(self,
            path,
            label_list,
            file_name = 'flavor_spectrum.xlsx',
            use_previous_batch = False):

        self.path = path
        self.label_list = label_list
        self.file_name = file_name
        self.use_previous_batch = use_previous_batch

        # 700-2500, resolution 2 from FOSS NIR system
        self.bandwidth = 900

        grabber = NIRInfoGrabber(path,
                label_list,
                file_name = file_name, 
                use_previous_batch = use_previous_batch)

        self.label_descriptions = grabber.catch_descriptions()

        self.data_dict, self.data_index = grabber.catch_info()

    def __len__(self):
        return len(self.data_dict)

    def spectrum_size(self):
        return (self.bandwidth)

    def dataset(self, select, index_list = None):
        select_list = ['agtron', 'flavor']
        if select.lower() not in select_list:
            raise ValueError(str(select) + 'is in invalid setting, \n' + str(select_list) + ' is available.')

        return self._build_batch_data(select, self.data_dict, index_list)

    def sample_spectrum(self, name, mode = 'mean', index = None):
        if mode not in ['mean', 'all', 'index']:
            raise ValueError("Argument: mode in NIRSpectrumDataset.sample_spectrum must be 'mean' or 'all'")

        spectrum = self.data_dict[self.data_index[name]]['spectrum']

        if mode.lower() == 'mean':
            spectrum = spectrum.mean(axis = 0)
        elif mode.lower() == 'index':
            if index is None:
                raise ValueError('Please set argument:index when using index mode.')

            if not isinstance(index, int):
                raise TypeError('Argument:index must be a int.')

            spectrum = spectrum[index]

        return spectrum

    def list_filenames(self):
        names = list(self.data_index.keys())
        names.sort()

        return names

    def _build_batch_data(self, select, data_dict, index_list):
        if index_list is None:
            index_list = list(self.data_dict.keys())

        spectra, labels = self._bind_data(select, data_dict, index_list)

        return spectra, labels

    def _bind_data(self, select, data_dict, index_list):
        select = select.lower()
        spectra, labels = [], []
        for i in range(len(index_list)):
            spectra.append(data_dict[index_list[i]]['spectrum'])
            labels.append(data_dict[index_list[i]][select])

        if len(spectra) > 0:
            spectra = np.concatenate(spectra, axis = 0)
            spectra = spectra.astype(np.float)
        else:
            spectra = np.array([[]])

        if len(labels) > 0:
            labels = np.concatenate(labels, axis = 0)
            labels = labels[:, : -1] # reduce the index column of data
            labels = labels.astype(np.int)
        else:
            labels = np.array([[]])

        return spectra, labels


