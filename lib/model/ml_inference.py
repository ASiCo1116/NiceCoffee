import logging

import numpy as np

from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import save_object, load_pickle_obj

from .base import DeepLearningModule
from .utils import TorchKernel
from .classifier import SVM

#--------------------------------------------------------------------------------
# ml_inference.py contains the inference part of some popular ML model, and it will
# be used in the ST gumbel-softmax estimator for probabilty estimation.
# The main objective of implementation is to make the ML model can be trace gradient by
# torch.autograd module.
#--------------------------------------------------------------------------------


__all__ = ['TorchSVC']


class TorchSVC(DeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 0,
            model_name = '',
            kernel = 'rbf',
            degree = 3,
            coef0 = 0.0,
            gamma = 1.0,
            batch = True,
            verbose = True):

        super(TorchSVC, self).__init__(
                batch = batch,
                verbose = verbose)

        if not isinstance(input_size, tuple):
            raise TypeError('Argumet: input_size only tuple is available.')

        if not isinstance(output_size, int):
            raise TypeError('Argument: output_size must be int.')

        if not isinstance(model_name, str):
            raise TypeError('Argument: model_name only str available.')

        kernel_available_list = ['linear', 'poly', 'sigmoid', 'rbf']
        if kernel.lower() not in kernel_available_list:
            raise ValueError('Argument: kernel must in ' + str(kernel_available_list))

        if not isinstance(degree, (int, float)):
            raise TypeError('Argument: degree must be a int or float.')

        if not isinstance(coef0, float):
            raise TypeError('Argument: coef0 must be a flaot.')

        if not isinstance(gamma, float):
            raise TypeError('Argument: gamma must be a float.')

        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.kernel = kernel.lower()

        self.train_mode = False

        self.kernel_function = TorchKernel(
                kernel = kernel,
                gamma = gamma,
                degree = degree,
                coef0 = coef0,
                support_vectors = None)

        if self.kernel_function.support_vectors is None:
            # check if the TorchSVC is only init for default
            if sum(input_size) < 1 or self.output < 1:
                self.weight = nn.Linear(1, 1)
            else:
                self.weight = nn.Linear(sum(input_size), output_size)
        else:
            n_supports = self.kernel_function.support_vectors.size()[0]
            self.weight = nn.Linear(n_supports, output_size)

    def forward(self, x, support_vectors = None):
        x = self.kernel_function(x)
        x = self.weight(x)

        return x

    def predict(self, x, support_vectors = None, return_type = 'origin'):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError('The TorchSVC model only accept numpy.ndarray or torch.Tensor object.')

        if isinstance(x, np.ndarray):
            origin_type = 'numpy'
            x = torch.from_numpy(x)
            device = self.weight.weight.device
            x = x.to(device)
        else:
            origin_type = 'torch'

        output = self.forward(x, support_vectors = support_vectors)
        output = (torch.sign(output) + 1.) / 2.
        output = output.int()

        if not self.batch:
            output = output.squeeze(dim = 1)

        if return_type == 'origin':
            output = self.transform_element_type(output, return_type = origin_type)
        else:
            output = self.transform_element_type(output, return_type = return_type)

        return output

    def load_from_ML(self, model, model_index = None):
        if model_index is None:
            raise TypeError('Please assign the model_index in the Concatenated SVM model.')

        if not isinstance(model, SVM):
            raise TypeError('Input model must be the SVM model trained by aroma_net.')

        self.input_size = model.input_size
        self.model_name = model.model_name + '(Model_index={0})'.format(model_index)
        self = self.load_from_sklearn(model.models[model_index])

        return self

    def load_from_sklearn(self, model):
        if not isinstance(model, SVC):
            raise TypeError('Input model of TorchSVC.load_from_sklearn must be the model of sklearn.svm.SVC object.')

        num_supports = model.n_support_
        if len(num_supports) != 2:
            raise RuntimeError('Now TorchSVC only supporting binary SVM.')

        kernel = model.kernel
        if model.kernel == 'rbf':
            gamma = model._gamma
            self.kernel_function = TorchKernel(
                     kernel = kernel,
                     gamma = gamma)

        elif model.kernel == 'poly':
            gamma, degree, coef0 = model._gamma, model.degree, model.coef0
            self.kernel_function = TorchKernel(
                     kernel = kernel,
                     gamma = gamma,
                     degree = degree,
                     coef0 = coef0)

        elif model.kernel == 'linear':
             self.kernel_function = TorchKernel(kernel = kernel)

        elif model.kernel == 'sigmoid':
             gamma, coef0 = model._gamma, model.coef0
             self.kernel_function = TorchKernel(
                     kernel = kernel,
                     gamma = gamma,
                     coef0 = coef0)

        sv = torch.from_numpy(model.support_vectors_)
        a = torch.from_numpy(model.dual_coef_)
        b = torch.from_numpy(model.intercept_)

        self.kernel_function.replace_support_vectors(sv)
        self.weight = nn.Linear(sum(num_supports), len(num_supports))
        self.weight.weight = nn.Parameter(a)
        self.weight.bias = nn.Parameter(b)

        return self

    def save(self, path):
        state = {}
        state['model_type'] = 'TorchSVC'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['kernel'] = self.kernel
        state['kernel_function'] = self.kernel_function
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = self.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: TorchSVC saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.TorchSVC only can load file endswith .adab which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'TorchSVC':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
               input_size = state['input_size'],
               output_size = state['output_size'],
               model_name = state['model_name'],
               kernel = state['kernel'])

        self.kernel_function = state['kernel_function']
        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self

    def __repr__(self):
        text = super(TorchSVC, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  kernel: ' + str(self.kernel) + ',\n'

        return text


