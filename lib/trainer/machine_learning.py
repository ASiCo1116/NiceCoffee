import os
import random
import logging

import numpy as np

from tqdm import tqdm

from ..data import GCMSClassifierDataset, NIRClassifierDataset
from ..eval import ClassificationReport
from ..model import model_from_config
from ..utils import transform_empty_argument

from .base import BaseTrainer
from .utils import CriterionProperty, init_directory
from .ratio_tuner import MultiClassSingleRatioTuner 

#--------------------------------------------------------------------------------
# machine_learning.py contains the extended object of lib.trainer.base.BaseTrainer
# the object is used in machine learning
#--------------------------------------------------------------------------------


__all__ = ['MachineLearningTrainer']


class MachineLearningTrainer(BaseTrainer):
    def __init__(self,
            config_path,
            datatype = 'gcms',
            directory_check = True):

        super(MachineLearningTrainer, self).__init__(
                config_path = config_path,
                directory_check = directory_check)

        if not isinstance(datatype, str):
            raise TypeError('Argument: datatype must be a string.')

        if datatype.lower() not in ['nir', 'gcms']:
            raise ValueError('Argument: datatype in lib.trainer.MachineLearningTrainer must be NIR or GCMS.')

        self.datatype = datatype
        self.increase_code_num = 3

    def eval(self, model, dataset):
        dataset.set_mode('train')
        train_data, train_label = dataset.catch_all()

        train_prediction = model.predict(train_data)

        train_report = ClassificationReport(train_prediction, train_label,
                label_note = dataset.descriptions())

        dataset.set_mode('val')
        val_data, val_label = dataset.catch_all()

        val_prediction = model.predict(val_data)

        val_report = ClassificationReport(val_prediction, val_label,
                label_note = dataset.descriptions())

        dataset.set_mode('test')
        test_data, test_label = dataset.catch_all()

        test_prediction = model.predict(test_data)

        test_report = ClassificationReport(test_prediction, test_label,
                label_note = dataset.descriptions())

        return train_report, val_report, test_report

    def _train(self, config = None):
        if config is None:
            config = self.config

        dataset = self._init_dataset(config)
        try:
            self.use_tuner = config['ratio_tuner_use']
        except KeyError:
            self.use_tuner = False

        if self.use_tuner:
            tuner_config = transform_empty_argument(config, 'ratio_tuner')
            self.ratio_tuner = MultiClassSingleRatioTuner(n_classes = 2, **tuner_config)

        model = self._init_model(config)
        model.train_mode = True

        criterion = config['criterion']
        repeat_time = config['eval_times']
        model = self._auto_tune_model(model, criterion, repeat_time, dataset)

        return model

    def _init_dataset(self, config = None):
        if config is None:
            config = self.config

        if self.datatype.lower() == 'nir':
            dataset = NIRClassifierDataset(**config['dataset'])
        elif self.datatype.lower() == 'gcms':
            dataset = GCMSClassifierDataset(**config['dataset'])

        dataset.summary_label_distribution()

        self.dataset = dataset

        return dataset

    def _init_model(self, config = None):
        if config is None:
            config = self.config

        model_object = model_from_config(config)
        model_config = transform_empty_argument(config['model'], 'argument')
        model = model_object(input_size = self.dataset.data_size(),
                output_size = self.dataset.descriptions(),
                verbose = False,
                **model_config)

        logging.info('Model information:')
        model.summary()

        return model

    def _record(self, config):

        train_report, val_report, test_report = self.eval(self.model, self.dataset)

        logging.info('\nTraining set:')
        train_report.summary()
        logging.info('\nValidation set:')
        val_report.summary()
        logging.info('\nTesting set:')
        test_report.summary()

        train_report.save(os.path.join(self.save_dir, 'train_set'))
        val_report.save(os.path.join(self.save_dir, 'validation_set'))
        test_report.save(os.path.join(self.save_dir, config['model_name']))

        return None

    def _auto_tune_model(self, model, criterion, repeat_time, dataset):
        for submodel_index in tqdm(range(len(model))):
            model = self._auto_tune_submodel(model, submodel_index, criterion, repeat_time, dataset)

        return model

    def _auto_tune_submodel(self, model, submodel_index, criterion, repeat_time, dataset):
        parameters = model.optimized_properties

        criterion_value = None
        optimized_parameters = self._initial_submodel_config(parameters)
        for para in list(parameters.keys()):
            optimized_parameters, criterion_value = self._tune_single_parameter(model, submodel_index,
                    criterion, repeat_time, optimized_parameters, parameters, para, dataset, criterion_value) 

            model = self._final_train(model, submodel_index, optimized_parameters, parameters, para, dataset)

        return model

    def _initial_submodel_config(self, parameters):
        optimized_parameter_config = {}
        for para in list(parameters.keys()):
            if parameters[para]['numeric']:
                optimized_parameter_config[para] = parameters[para]['lower_bound']
            else:
                optimized_parameter_config[para] = paramters[para]['available'][0]

        return optimized_parameter_config

    def _tune_single_parameter(self, model, submodel_index, criterion, repeat_time, \
            optimized_parameters, parameters, para, dataset, criterion_value):

        if parameters[para]['numeric']:
            optimized_parameters, criterion_value = self._tune_numeric(model, submodel_index,
                    criterion, repeat_time, optimized_parameters, parameters, para, dataset,
                    criterion_value)

        else:
            optimized_parameters, criterion_value = self._tune_none_numeric(model, submodel_index,
                    criterion, repeat_time, optimized_parameters, parameters, para, dataset,
                    criterion_value)

        return optimized_parameters, criterion_value

    def _tune_numeric(self, model, submodel_index, criterion, repeat_time, \
            optimized_parameters, parameters, para, dataset, criterion_value):

        # record the initial predicted state
        # exp_binary is the expoential binary index of for searching value

        if criterion_value is None:
            criterion_value = self._model_repeat_predict(model, submodel_index, criterion, repeat_time,
                    optimized_parameters, dataset)
        else:
            criterion_value = value

        exp_binary = self._initial_binary_list(parameters, para)

        try:
            increase_code_num = parameters[para]['binary_digit_overflow']
        except KeyError:
            increase_code_num = self.increase_code_num

        increase_code_list = [True for i in range(increase_code_num)]
        increase_check_index, index = 0, 0
        while True:
            tmp_parameters = optimized_parameters.copy()

            exp_mode = False
            for element in increase_code_list:
                if element:
                    exp_mode = True

            if exp_mode:
                exp_binary.insert(0, 1)
                test_value = self._transform_binary(exp_binary)

                tmp_parameters[para] = parameters[para]['type_function'](test_value)

                new_criterion_value = self._model_repeat_predict(model, submodel_index, criterion, repeat_time, 
                        optimized_parameters, dataset)

                if CriterionProperty[criterion] == 'high':
                    if new_criterion_value >= criterion_value:
                        optimized_parameters = tmp_parameters
                        criterion_value = new_criterion_value
                        increase_check_index = 0
                        increase_code_list = [True for i in range(increase_code_num)]
                        index = 0
                    else:
                        increase_code_list[increase_check_index] = False
                        increase_check_index += 1
                        index += 1

                elif CriterionProperty[criterion] == 'low':
                    if new_criterion_value <= criterion_value:
                        optimized_parameters = tmp_parameters
                        criterion_value = new_criterion_value
                        increase_check_index = 0
                        increase_code_list = [True for i in range(increase_code_num)]
                        index = 0
                    else: 
                        increase_code_list[increase_check_index] = False
                        increase_check_index += 1
                        index += 1

                if test_value > parameters[para]['upper_bound']:
                    increase_code_list = [False for i in range(increase_code_num)]
                    index = 0

            else:
                if index >= len(exp_binary):
                    break

                tmp_exp_binary = exp_binary.copy()
                if tmp_exp_binary[index] == 1:
                    tmp_exp_binary[index] = 0
                elif tmp_exp_binary[index] == 0:
                    tmp_exp_binary[index] = 1

                test_value = self._transform_binary(tmp_exp_binary)
                tmp_parameters[para] = parameters[para]['type_function'](test_value)

                new_criterion_value = self._model_repeat_predict(model, submodel_index, criterion, repeat_time,
                        optimized_parameters, dataset)

                if CriterionProperty[criterion] == 'high':
                    if new_criterion_value >= criterion_value:
                        exp_binary = tmp_exp_binary
                        optimized_parameters = tmp_parameters
                        criterion_value = new_criterion_value

                elif CriterionProperty[criterion] == 'low':
                     if new_criterion_value <= criterion_value:
                        exp_binary = tmp_exp_binary
                        optimized_parameters = tmp_parameters
                        criterion_value = new_criterion_value

                index += 1

        return optimized_parameters, criterion_value

    def _transform_binary(self, exp_binary_list):
        return np.sum(np.power(2., np.arange(len(exp_binary_list))[:: -1]) * exp_binary_list)

    def _initial_binary_list(self, parameters, para):
        value = parameters[para]['lower_bound']
        value = bin(int(value))[2: ]
        transformed_list = []
        for i in range(len(value)):
             transformed_list.append(int(value[i]))

        return transformed_list

    def _tune_none_numeric(self, model, submodel_index, criterion, repeat_time, \
            optimized_parameters, parameters, para, dataset, criterion_value):

        # the logic flow of combining numeric and none-umeric parameters is not determined
        # therefore, the none-numeric parameters was recommended to be default set in the
        # model class for a more accurate model tuning process

        if criterion_value is None:
            criterion_value = self._model_repeat_predict(model, submodel_index, criterion, repeat_time,
                    optimized_parameters, dataset)
        else:
            criterion_value = value

        available_list = parameters[para]['available']
        for var in available_list:
            tmp_parameters = optimized_parameters.copy()
            tmp_parameters[para] = var

            new_criterion_value = self._model_repeat_predict(model, submodel_index, criterion, repeat_time, 
                    optimized_parameters, dataset)

            if CriterionProperty[criterion] == 'high':
                if new_criterion_value >= criterion_value:
                    optimized_parameters = tmp_parameters
                    criterion_value = new_criterion_value

            elif CriterionProperty[criterion] == 'low':
                if new_criterion_value <= criterion_value:
                    optimized_parameters = tmp_parameters
                    criterion_value = new_criterion_value

        return optimized_parameters, criterion_value

    def _model_repeat_predict(self, model, submodel_index, criterion, repeat_time, \
            optimized_parameters, dataset):

        criterion_value = []
        for i in range(repeat_time):
            dataset.resplit()
            dataset.set_mode('train')
            train_data, train_label = dataset.catch_all()
            dataset.set_mode('val')
            val_data, val_label = dataset.catch_all()

            if self.use_tuner:
                temp_data, temp_label = self.ratio_tuner.tune(train_data, train_label[:, submodel_index])
            else:
                temp_data, temp_label = train_data, train_label[:, submodel_index]


            model.train(submodel_index, temp_data, temp_label,
                    **optimized_parameters)

            prediction = model.single_predict(val_data, index = submodel_index)
            criterion_value.append(ClassificationReport(prediction.reshape(-1, 1), \
                   val_label[:, submodel_index].reshape(-1, 1), info = False).get_index(criterion))

        criterion_value = np.mean(np.array(criterion_value))

        return criterion_value

    def _final_train(self, model, submodel_index, optimized_parameters, parameters, para, dataset):
        dataset.set_mode('train')
        train_data, train_label = dataset.catch_all()
        dataset.set_mode('val')
        val_data, val_label = dataset.catch_all()

        data = np.concatenate((train_data, val_data), axis = 0)
        label = np.concatenate((train_label, val_label), axis = 0)

        if self.use_tuner:
            temp_data, temp_label = self.ratio_tuner.tune(train_data, train_label[:, submodel_index])
        else:
            temp_data, temp_label = train_data, train_label[:, submodel_index]

        model.train(submodel_index, temp_data, temp_label,
                **optimized_parameters)

        return model


