import abc
import logging

import numpy as np

import torch
import torch.nn as nn

from ..data.transforms import TransformFunctionObject
from ..utils import save_object, load_pickle_obj

#--------------------------------------------------------------------------------
# base.py contains the API format of each framework
#--------------------------------------------------------------------------------


__all__ = ['BaseModule', 'MachineLearningModule', 'DeepLearningModule',
        'ConcatenatedML', 'AllZeroModel']


class BaseModule(abc.ABC):
    def __init__(self):
        self.train_mode = False

    def summary(self):
        text = self.__repr__()
        logging.info(text)
        return None

    def transform_element_type(self, element, return_type = 'origin'):
        if return_type.lower() not in ['origin', 'numpy', 'torch']:
            raise ValueError('Argument: return_type must be str:origin or str:numpy or str:torch.')

        if isinstance(element, np.ndarray):
            if return_type.lower() == 'torch':
                element = torch.tensor(element)
        elif isinstance(element, torch.Tensor):
            if return_type.lower() == 'numpy':
                element = element.detach().cpu().numpy()

        return element

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, path):
        raise NotImplementedError()

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError()


class MachineLearningModule(BaseModule):
    def __init__(self):
        super(MachineLearningModule, self).__init__()


class DeepLearningModule(nn.Module, BaseModule):
    def __init__(self,
            batch = True,
            verbose = False):

        super(DeepLearningModule, self).__init__()

        self.preprocess_func = None

        if not isinstance(batch, bool):
            raise TypeError('Argument: batch must be a boolean, please input True or False for accept batch mode.')

        if not isinstance(verbose, bool):
            raise TypeError('Argument: verbose must be a boolean, actually this argument is useless in DeepLearningModule.')

        self.batch = batch
        self.verbose = False

    def predict(self, *input, return_type = 'origin'):
        # if not self.train_mode:
        # input = self._preprocess(*input)
        input = torch.tensor(np.expand_dims(input, axis=0)).float()
        output = self.forward(*input)

        if isinstance(output, (list, tuple)):
            final_output = []
            for element in output:
                final_output.append(self.transform_element_type(element))

            if isinstance(output, tuple):
                final_output = tuple(final_output)
        else:
            output = self.transform_element_type(output)

        return output

    def count_params(self):
        paras_num = sum(p.numel() for p in self.parameters())
        return paras_num

    def set_preprocess(self, preprocess):
        if preprocess is not None:
            if not isinstance(preprocess, TransformFunctionObject):
                raise TypeError('Preprocess set in the DL model must be the object in lib.data.transforms.')

        self.preprocess_func = preprocess

        return None

    def _preprocess(self, *input):
        if self.preprocess_func is not None:
            raise NotImplementedError('self._preprocess should be defined if preprocess function is apply.')

        return input


class AllZeroModel(MachineLearningModule):
    def __init__(self):
        super(MachineLearningModule, self).__init__()

    def save(self, path):
        state = {}

        state['model_type'] = 'AllZeroModel'

        save_object(path, state, mode = 'pickle', extension = 'zmd')
        logging.info('Model: AllZeroModel saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.zmd'):
            raise RuntimeError('lib.model.AllZeroModel only can load file endswith .zmd which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        logging.info('Load state file success !!')

        return None

    def predict(self, data, return_type = 'origin'):
        sample_num = data.shape[0]
        outputs = np.zeros((sample_num))

        outputs = self.transform_element_type(outputs, return_type = return_type)

        return outputs

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ensemble concatenated classifer for multiple label
class ConcatenatedML(MachineLearningModule):
    def __init__(self,
            input_size,
            output_size,
            model_name = '',
            batch = True,
            verbose = True):

        super(ConcatenatedML, self).__init__()

        if not isinstance(input_size, tuple):
            raise TypeError('Argumet: input_size only tuple is available.')

        if isinstance(output_size, int):
            self.output_size = output_size
            self.output_descriptions = None
        elif isinstance(output_size, (list, tuple)):
            self.output_size = len(output_size)
            self.output_descriptions = output_size
        else:
            raise TypeError('Only description list or int can be fed into model.')

        if not isinstance(model_name, str):
            raise TypeError('Argument: model_name only str available.')

        if not isinstance(batch, bool):
            raise TypeError('Argument: batch must be a boolean, please input True or False for accept batch mode.')

        if not isinstance(verbose, bool):
            raise TypeError('Argument: verbose must be a boolean, please input True or False for accept batch mode.')


        self.input_size = input_size
        self.model_name = model_name
        self.batch = batch
        self.verbose = verbose

        self.preprocess_func = None
        self.optimized_properties = {}

    def __len__(self):
        return len(self.models)

    def train(self, index, data, label, preprocess = False, **kwargs):
        data = self._check_input(data)
        if preprocess:
            data = self._preprocess(data)

        self._train(index, data, label, **kwargs)

        if self.verbose:
            logging.info('Finish training model (index ' + str(index) + ' ).')

        return None

    def single_predict(self, data, index = 0):
        data = self._check_input(data)
        data = self._preprocess(data)

        output_size = (data.shape[0], 1)
        output = np.zeros(output_size)
        output[:, 0] = self.models[index].predict(data)

        return output

    def set_preprocess(self, preprocess):
        if preprocess is not None:
            if not isinstance(preprocess, TransformFunctionObject):
                raise TypeError('Preprocess set in the ML concatenated model must be the object in lib.data.transforms.')

        self.preprocess_func = preprocess

        return None

    def predict(self, data, return_type = 'origin'):
        if not self.train_mode:
            data = self._preprocess(data)
            data = self._check_input(data)

        output_size = (data.shape[0], self.output_size)
        output = np.zeros(output_size)
        for i in range(len(self.models)):
            output[:, i] = self.models[i].predict(data)

        output = self.transform_element_type(output, return_type = return_type)

        return output

    def _check_input(self, data):
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError('The ML based model only accept numpy.ndarray or torch.Tensor object.')
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self.batch:
            data_size = data.shape[1: ]
        else:
            data_size = data.shape
            data = np.expand_dims(data, axis = 0)

        if data_size != self.input_size:
            print(data_size, self.input_size)
            raise RuntimeError('Size mismatch: input data must be in size: ' + str(self.input_size))

        return data

    def _preprocess(self, data):
        if self.preprocess_func is not None:
            data = self.preprocess_func(data)

        return data


