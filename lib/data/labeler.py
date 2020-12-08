import os
import abc
import csv
import json
import logging

import numpy as np
import pandas as pd

from ..utils import *
from .utils import NIRExcelInfo, select_nir_label_list

#--------------------------------------------------------------------------------
# labeler.py contains object which can grab label information from flavor_list.xlsx
# flavor_list.xlsx contains raw tag information, details of each coffee would not
# push to the repository.
#--------------------------------------------------------------------------------


__all__ = ['BaseInfoGrabber', 'GCMSLabelInfoGrabber', 'NIRInfoGrabber']


class BaseInfoGrabber(abc.ABC):
    @abc.abstractmethod
    def catch_info(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def catch_descriptions(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def statisticize_labels(self):
        raise NotImplementedError()


class GCMSLabelInfoGrabber(BaseInfoGrabber):
    def __init__(self, 
            file_path,
            path = 'object'):

        super(GCMSLabelInfoGrabber, self).__init__()

        if not isinstance(file_path, str):
            raise TypeError('Arguemnt: file_path must be a string.')

        wheel_path = os.path.join(path,'flavor_wheel.json')
        self.wheel_info = load_json_obj(wheel_path)

        label_paths = {'inner': os.path.join(path, 'inner_label.json'),
                'middle': os.path.join(path, 'middle_label.json'),
                'outer': os.path.join(path, 'outer_label.json')}
        self.labels = self._load_dict_json(label_paths)

        self.file_path = os.path.join(file_path, 'flavor_list.xlsx')
        self.info_name = ['fragrance', 'aroma']

        # data are pandas.dataframe
        self.data = {'fragrance': self._read_excel(self.file_path, 'fragrance'),
                'aroma': self._read_excel(self.file_path, 'aroma')}

    def catch_info(self, select, mode = 'multi-hot'):
        if select not in ['fragrance', 'aroma', 'all']:
            raise ValueError(select, ' must be fragrance or aroma.')

        if mode not in ['multi-hot']:
            raise ValueError(mode, ' is not a valid catch_labels mode.')

        if mode == 'multi-hot':
            if select == 'all':
                return self.data
            else:
                return self.data[select]

    def catch_descriptions(self, info_name):
        if info_name not in self.info_name:
            raise ValueError(info_name, 'is not in valid acquired information.')

        # pandas.dataframe default save data as type: numpy.ndarray
        flavor_list = self.data[info_name].columns.values.tolist()
        flavor_list = self._clean_excel_header_space(flavor_list)
        return flavor_list

    def list_descriptions(self, info_name, mode = 'change_line'):
        flavor_list = self.catch_descriptions(info_name)
        logging.info(info_name + ':')
        self._print_list(flavor_list, mode)
        return None

    def statisticize_labels(self):
        statistic_table = self._build_statistic_table()
        for selected in statistic_table.keys():
            self._print_statistic_table_summary(selected, statistic_table[selected])

        return None

    def _print_statistic_table_summary(self, name, table):
        smell = name.split('/')[0]
        part = name.split('/')[1]

        selected_wheel_list = self.wheel_info[part]
        summation = np.sum(table, axis = 0)
        assert summation.shape[0] == len(selected_wheel_list)

        logging.info('=' * 50)
        logging.info('Summary of table:' + name)
        logging.info('Total number:' + str(table.shape[0]) + '\n')

        for i in range(len(selected_wheel_list)):
            logging.info(str(selected_wheel_list[i]) + ':' + str(summation[i]))

        logging.info('=' * 50)
        return None

    def _build_statistic_table(self):
        statistic_table = {}
        for smell in self.info_name:
            dataframe = self.data[smell]
            for part in self.labels.keys():
                #create the statistic table for spcific smell and wheel part
                statistic_table[smell + '/' + part] = []

                # check every samples in the dataframe
                sample_names = dataframe.index.tolist()
                for index in range(len(sample_names)):
                    # create record list for each sample
                    single_sample_table = [0.0 for i in range(len(self.wheel_info[part]))]
                    for check_label in dataframe.columns.values.tolist():
                        if check_label == 'sample name':
                            continue

                        if dataframe.loc[sample_names[index], check_label] > 0.0:
                            # flavor exists, therfore, grab the index of the record list
                            label_tag_indexs = self._grab_label_tag_index(check_label, part)
                            if len(label_tag_indexs) != 0:
                                for i in range(len(label_tag_indexs)):
                                    single_sample_table[label_tag_indexs[i]] = 1.0
                            else:
                                pass
                                logging.info('Exist a exception flavor:' + str(check_label) + 
                                        'did not in the label, please add new tags into labels.json.')
                        else:
                            pass

                    # insert the outcome to the statistic table
                    statistic_table[smell + '/' + part].append(single_sample_table)

                # convert the table from list to numpy array
                statistic_table[smell + '/' + part] = np.asarray(statistic_table[smell + '/' + part])

        return statistic_table

    def _grab_label_tag_index(self, exist_description, label_part):
        arranged_index_dict = self.wheel_info[label_part]
        index = 0
        indexs = []
        for select_flavor in arranged_index_dict:
            if exist_description in self.labels[label_part][select_flavor]:
                indexs.append(index)
                index += 1
            else:
                index += 1

        return indexs

    def _clean_excel_header_space(self, the_list):
        for obj in the_list:
            if 'Unnamed' in obj:
                the_list.remove(obj)

        return the_list

    def _print_list(self, the_list, mode):
        if mode not in ['origin', 'change_line']:
            raise ValueError(mode, 'is not in implemented print method.')

        if mode == 'change_line':
            for obj in the_list:
                logging.info(obj)
        elif mode == 'origin':
            logging.info(the_list)

        logging.info('')
        return None

    def _load_dict_json(self, path_dict):
       data_dict = {}
       for key in path_dict.keys():
           data_dict[key] = load_json_obj(path_dict[key])

       return data_dict

    def _read_excel(self, path, sheet_name):
        if not path.endswith('.xlsx'):
            raise RuntimeError(path, ' is not a excel file')

        # non-fill in space would fill in float zero
        dataframe = pd.read_excel(path, sheet_name = sheet_name)
        dataframe = dataframe.fillna(value = 0.0)
        dataframe.set_index('sample name', inplace = True)

        return dataframe


class NIRInfoGrabber(BaseInfoGrabber):
    def __init__(self,
            file_path,
            label_list,
            file_name = 'flavor_spectrum.xlsx',
            use_previous_batch = False):

        super(NIRInfoGrabber, self).__init__()

        if not isinstance(file_path, str):
            raise TypeError('Arguemnt: file_path must be a string.')

        if not isinstance(file_name, str):
            raise TypeError('Arguemnt: file_name must be a string.')

        self.file_path = file_path
        self.file_name = file_name
        self.band_width = 900
        self.previous_index_start = 124
        self.use_previous_batch = use_previous_batch

        self.excel_path = os.path.join(file_path, file_name)
        self.label_list = select_nir_label_list(label_list)
        self.info = NIRExcelInfo

        self.dataframe = pd.read_excel(self.excel_path, '工作表1')
        self.sample_name, self.spectrum, self.label, self.agtron = self._from_dataframe(self.dataframe,
                self.label_list)

        self.data, self.data_index = self._rebuild_data_dict(self.sample_name, self.spectrum, self.label, self.agtron)

    def catch_info(self):
        return self.data, self.data_index

    def catch_descriptions(self):
        return self.label_list[: -1]

    def statisticize_labels(self, select = 'flavor'):
        if select == 'flavor':
            summation = np.sum(self.lable, axis = 0)
            assert summation.shape[0] == (len(self.label_list) - 1)

            logging.info('=' * 50)
            logging.info('Summary of label:')

            for i in range((len(self.label_list) - 1)):
                logging.info(str(self.label_list[i]) + ': ' + str(summation[i]))

            logging.info('=' * 50)
            return None

        elif select == 'agtron':
            # may finish or not ?
            pass

        else:
            raise ValueError(select, ' is not available in th NIRInfoGrabber.statisticize_labels().')

    def _from_dataframe(self, dataframe, label_list):
        sample_name = []

        if self.use_previous_batch:
            reduce_num = self.previous_index_start
        else:
            reduce_num = 0

        spectrum = np.empty([len(dataframe) - reduce_num, self.band_width])
        label = np.empty([len(dataframe) - reduce_num, len(label_list)])
        agtron = np.empty([len(dataframe) - reduce_num])

        if self.use_previous_batch:
            start = 0
        else:
            # the index of new version standard excel, please check the data version
            start = self.previous_index_start

        iter_index = 0
        for i in range(start, len(dataframe)):
            sample_name.append(dataframe.iloc[i, self.info['Sample Number'][0]: self.info['Sample Number'][1]])
            spectrum[iter_index] = np.array(dataframe.iloc[i, self.info['Spectrum'][0]: self.info['Spectrum'][1]])
            agtron[iter_index] = np.array(dataframe.iloc[i, self.info['Agtron_Ground'][0]: self.info['Agtron_Ground'][1]])

            for j in range(len(label_list)):
                if self.info[label_list[j]] == None:
                    #index part
                    label[iter_index, j] = i
                else:
                    label[iter_index, j] = self._check_repeat(dataframe.iloc[i,
                        self.info[label_list[j]][0]: self.info[label_list[j]][1]])

            iter_index += 1

        return sample_name, spectrum, label, agtron

    def _check_repeat(self, part_dataframe):
        check_array = np.array(part_dataframe)
        if np.sum(part_dataframe) >= 1:
            return 1
        else:
            return 0

    def _rebuild_data_dict(self, name_list, spectrum, label, agtron):
        count, iter_index = 0, 0
        data_dict = {}
        index_dict = {}
        while iter_index < len(name_list):
            if self.use_previous_batch and count < self.previous_index_start:
                data_dict[count] = {}
                data_dict[count]['name'] = name_list[iter_index].iloc[0]
                data_dict[count]['index'] = [iter_index] 
                data_dict[count]['spectrum'] = spectrum[iter_index].reshape(-1, self.band_width)
                data_dict[count]['label'] = label[iter_index].reshape(-1, len(self.label_list))
                data_dict[count]['agtron'] = agtron[iter_index]
                index_dict[name_list[iter_index].iloc[0]] = count
                count += 1
                iter_index += 1
            else:
                if name_list[count].iloc[0].endswith('G'):
                    data_dict[count] = {}
                    data_dict[count]['name'] = name_list[iter_index].iloc[0]
                    data_dict[count]['index'] = [iter_index]
                    data_dict[count]['spectrum'] = spectrum[iter_index].reshape(-1, self.band_width)
                    data_dict[count]['flavor'] = label[iter_index].reshape(-1, len(self.label_list))
                    data_dict[count]['agtron'] = np.array([1, agtron[iter_index]])
                    index_dict[name_list[iter_index].iloc[0]] = count
                    count += 1
                    iter_index += 1
                else:
                    data_dict[count] = {}
                    data_dict[count]['name'] = name_list[iter_index].iloc[0]
                    data_dict[count]['index'] = [iter_index, iter_index + 1, iter_index + 2]
                    data_dict[count]['spectrum'] = spectrum[iter_index: iter_index + 3].reshape(-1, self.band_width)
                    data_dict[count]['flavor'] = label[iter_index: iter_index + 3].reshape(-1, len(self.label_list))
                    data_dict[count]['agtron'] = np.array([agtron[iter_index] for i in range(3)]).reshape(1, 3)
                    index_dict[name_list[iter_index].iloc[0][: -2]] = count
                    count += 1
                    iter_index += 3

        return data_dict, index_dict


