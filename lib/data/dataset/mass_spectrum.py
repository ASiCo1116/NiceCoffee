import os
import logging
import numpy as np
import pandas as pd

import gcmstools
from gcmstools.filetypes import AiaFile

from ..labeler import GCMSLabelInfoGrabber

#--------------------------------------------------------------------------------
# mass_spectrum_dataset.py contains the object which can search data directory
#--------------------------------------------------------------------------------


__all__ = ['MassSpectrumDataset']


class MassSpectrumDataset(object):
    def __init__(self,
            path,
            datatype = '2d',
            search = False):

        # set the directory which grab the data
        # set the label file 
        self.path = path
        labeler = GCMSLabelInfoGrabber(file_path = path)
        self.labels = labeler.catch_info('all')
        self.label_descriptions = {'aroma': labeler.catch_descriptions('aroma'),
                'fragrance': labeler.catch_descriptions('fragrance')}

        # set datatype, 1d is tic, 2d is mass spectrum
        if datatype.lower() not in ['1d', '1d_rts', '1d_mzs', '2d']:
            raise ValueError(str(datatype) + 'is invalid setting for MassSpectrumDataset.datatype, [1d (equals to 1d_MZS), 1d_RTS (sum at retention time axis), 1d_MZS (sum at M/Z axis) or 2d].')

        if not isinstance(search, bool):
            raise TypeError('Argument: search must be a boolean [True or False].')

        self.datatype = datatype
        self.file_extension = '.CDF'
        self.blank_mark = 'BLANK-SPME'
        self.mass_length = 500
        self.time_length = 5000
        self.search = search 
        self.search = self._check_saved_config(path, self.search)

        if self.search:
            logging.info('Start to collect .CDF files in folder:' + str(path))
            self.aromas, self.fragrances, self.blank_files, self.blank_table = [], [], [], {} 
            self._search_directory(self.aromas, self.fragrances, self.blank_files, self.blank_table, path,
                    self.file_extension, self.blank_mark)
            self._write_config(path, self.aromas, self.fragrances, self.blank_files, self.blank_table)
        else:
            self.aromas, self.fragrances, self.blank_files, self.blank_table = self._read_from_config(path)

        logging.info('Start to build dataset from read files ...')
        self.samples = {'aroma': self._build_cdf(self.aromas), 'fragrance': self._build_cdf(self.fragrances)}
        self.blank_samples = self._build_cdf(self.blank_files)

    def __len__(self):
        return len(self.aromas) + len(self.fragrances) + len(self.blank_files)

    def spectrum_size(self):
        return (self.time_length, self.mass_length)

    def dataset(self, select, reduce_blank = True):
        select_list = ['fragrance', 'aroma']
        if select not in select_list:
            raise ValueError(str(select) + 'is in invalid setting, \n' + str(select_list) + ' is available.')

        return self._build_batch_data(select, self.samples[select]['data'],
                self.samples[select]['table'], reduce_blank)

    def sample_spectrum(self, name):
        if name.endswith('_Dry'):
            return self.samples['fragrance']['data'][name]
        elif name.endswith('_Wet'):
            return self.samples['aroma'][name]

        if datatype == 'aroma':
            data = self.samples['aroma']['data'][name + '_Wet']
        elif datatype == 'fragrance':
            data = self.samples['fragrance']['data'][name + '_Dry']
        else:
            try:
                data = self.blank_samples['data'][name]
            except KeyError:
                raise ValueError(name, ' can not be found in blank files, please use datatype to assign set.')

        return data

    def list_filenames(self, blank = False):
        if blank:
            all_files = list(self.samples['fragrance']['data'].keys()) + \
                    list(self.samples['aroma']['data'].keys()) + \
                    list(self.blank_samples['data'].keys())

        else:
            all_files = list(self.samples['fragrance']['data'].keys()) + \
                    list(self.samples['aroma']['data'].keys())

        all_files.sort()

        return all_files

    def _check_saved_config(self, path, search, filename = 'samples.config'):
        if os.path.isfile(os.path.join(path, filename)) and not search:
            logging.info('Read the dataet history config, file searching process will not execute.')
            return False
        else:
            if not search:
                logging.info('Can not detect dataset config, forcibly execute searching process.')
                return True

        return search

    def _write_config(self, path, aromas, fragrances, blanks, table, filename = 'samples.config'):
        filename = os.path.join(path, filename)
        lines = ['fragrances:\n']
        for f in fragrances:
            lines.append(f + ',' + table[f] + '\n')

        lines.append('aromas:\n')
        for f in aromas:
            lines.append(f + ',' + table[f] + '\n')

        lines.append('blanks:\n')
        for f in blanks:
            lines.append(f + '\n')

        with open(filename, 'w') as f:
            f.writelines(lines)
            f.close()
            logging.info('Updated dataset config file was successfully writen in ' + str(filename))

        return None

    def _read_from_config(self, path, filename = 'samples.config'):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()
            f.close()

        mode = None
        aromas, fragrances, blanks, table = [], [], [], {}
        for line in lines:
            line = line.replace('\n', '')
            if 'fragrances:' in line:
                mode = 'fragrance'
                continue
            elif 'aromas:' in line:
                mode = 'aroma'
                continue
            elif 'blanks:' in line:
                mode = 'blank'
                continue

            files = line.split(',')
            if mode == 'fragrance':
                fragrances.append(files[0])
                table[files[0]] = files[1]
            elif mode == 'aroma':
                aromas.append(files[0])
                table[files[0]] = files[1]
            elif mode == 'blank':
                blanks.append(files[0])
            else:
                raise RuntimeError('The config files is not in correct format, please delete it and renew the dataset.')

        return aromas, fragrances, blanks, table

    def _build_batch_data(self, select, data_dict, table, reduce_blank):
        # blank is the arguement for checking the data_dict is not blank samples
        mass_spectra, ordered_keys = self._bind_dict(data_dict, table, reduce_blank)
        labels = self._bind_labels(select, ordered_keys)

        return mass_spectra, labels, ordered_keys, self.label_descriptions[select]

    def _bind_labels(self, select, keys):
        df = self.labels[select]
        label_array = []
        for key in keys:
            key = os.path.split(key)[-1].replace(self.file_extension, '')[: -4]
            label_array.append(np.array(df.loc[key]))

        label_array = np.stack(label_array, axis = 0)

        return label_array

    def _bind_dict(self, data_dict, table, reduce_blank):
        if len(data_dict) <= 0:
            raise RuntimeError('There is not any mass spectrum file in the folder, bind data array failed !!')

        keys = list(data_dict.keys())
        keys.sort()

        if self.datatype.lower() == '1d' or self.datatype.lower() == '1d_mzs':
            shape = (len(data_dict), self.time_length)
        elif self.datatype.lower() == '1d_rts':
            shape = (len(data_dict), self.mass_length)
        elif self.datatype == '2d':
            shape = (len(data_dict), self.time_length, self.mass_length)

        data_array = np.empty(shape)
        for i in range(len(keys)):
            if self.datatype == '1d' or self.datatype.lower() == '1d_mzs':
                data = data_dict[keys[i]][:self.time_length]
            elif self.datatype.lower() == '1d_rts':
                data = data_dict[keys[i]][:self.mass_length]
            elif self.datatype == '2d':
                data = data_dict[keys[i]][:self.time_length, :self.mass_length]

            if reduce_blank:
                if len(self.blank_table[table[keys[i]]]) != 0:
                    filename = os.path.split(self.blank_table[table[keys[i]]])[-1].replace(self.file_extension, '')
                    if self.datatype == '1d' or self.datatype.lower() == '1d_mzs':
                        blank = self.blank_samples['data'][filename][:self.time_length]
                    elif self.datatype.lower() == '1d_rts':
                        blank = self.blank_samples['data'][filename][:self.mass_length]
                    elif self.datatype == '2d':
                        blank = self.blank_samples['data'][filename][:self.time_length, :self.mass_length]

                    data -= blank
                    data[data < 0.] = 0.

            data_array[i] = data

        return data_array, keys

    def _search_directory(self, aromas, fragrances, blank_files, blank_table, path, file_extension, blank_mark):
        # recursively search the mass spectrum in directroy
        unfold_dir = os.listdir(path)
        blank_filename = [s for s in unfold_dir if blank_mark in s]
        if len(blank_filename) > 1:
            raise RuntimeError(path, 'contains more than one blank in the folder.')

        if len(blank_filename) == 0:
            logging.warning(path + ' do not have blank files.')

        for obj in unfold_dir:
            if os.path.isdir(os.path.join(path, obj)):
                # obj is a directory, keep search
                self._search_directory(aromas, fragrances, blank_files, blank_table,
                        os.path.join(path, obj), file_extension, blank_mark)
            else:
                if obj.endswith(file_extension):
                    if 'blank' in obj.lower():
                        blank_files.append(os.path.join(path, obj))

                    else:
                        if len(blank_filename) > 0:
                            blank_table[os.path.join(path, obj)] = os.path.join(path, blank_filename[0])
                        else:
                            blank_table[os.path.join(path, obj)] = ''

                        if obj.replace(file_extension, '').lower().endswith('dry'):
                            fragrances.append(os.path.join(path, obj))

                        elif obj.replace(file_extension, '').lower().endswith('wet'):
                            aromas.append(os.path.join(path, obj)) 

                        else:
                            logging.warning(os.path.join(path, obj) + \
                                    ' can not be recognized as aroma or fragrance files, skip adding to file list.')

        return None

    def _build_cdf(self, files_path_list):
        data = {}
        table = {}
        for file in files_path_list:
            temp = AiaFile(file)
            name = os.path.split(temp.filename)[-1].replace(self.file_extension, '')
            table[name] = file
            if self.datatype.lower() == '1d':
                data[name] = np.sum(temp.intensity, axis = 1)
            elif self.datatype.lower() == '1d_mds':
                data[name] = np.sum(temp.intensity, axis = 1)
            elif self.datatype.lower() == '1d_rts':
                data[name] = np.sum(temp.intensity, axis = 0)
            elif self.datatype.lower() == '2d':
                data[name] = temp.intensity

            else:
                raise RuntimeError('MassSpectrumDataset can not grab data from cdf files.')

            if 'BLANK-system' not in file:
                if len(temp.masses) < self.mass_length:
                    self.mass_length = len(temp.masses)

                if len(temp.times) < self.time_length:
                    self.time_length = len(temp.times)

        return {'data': data, 'table': table}


