import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#--------------------------------------------------------------------------------
# utils.py contains the function which will be used in aroma_net.model module
#--------------------------------------------------------------------------------


__all__ = ['EmptyFunction', 'select_conv', 'select_norm', 'conv_argument_transform',
        'PermuteLayer', 'RNNPermuteLayer', 'TorchKernel']


def EmptyFunction(input_object):
    return input_object

def select_conv(dim, in_channel, out_channel, conv_fsize, stride):
    if dim.lower() == '1d':
        return nn.Conv1d(in_channel, out_channel, kernel_size = conv_fsize,
                stride = stride, padding = conv_fsize // 2, bias = False)
    elif dim.lower() == '2d':
        padding = []
        for fsize in conv_fsize:
            padding.append(fsize // 2)

        padding = tuple(padding)

        return nn.Conv2d(in_channel, out_channel, kernel_size = conv_fsize,
                stride = stride, padding = padding, bias = False)

def select_norm(dim, num_feature, select = 'batch'):
    if select not in ['batch', 'instance']:
        raise ValueError(select, ' is not valid for function: select_norm.')

    if dim not in ['1d', '2d']:
        raise ValueError(dim, ' is not valid for function: select_norm.')

    if select == 'batch':
        if dim == '1d':
            return nn.BatchNorm1d(num_feature)
        elif dim == '2d':
            return nn.BatchNorm2d(num_feature)

    if select == 'instance':
        if dim == '1d':
            return nn.InstanceNorm1d(num_feature)
        elif dim == '2d':
            return nn.InstanceNorm2d(num_feature)

def conv_argument_transform(argument, dim):
    if not isinstance(argument, (int, tuple, list)):
        raise TypeError('Argument: fsize must be a int or tuple or list.')
            
    if dim == '1d':
        if isinstance(argument, (tuple, list)):
            raise RuntimeError('List or tuple conv_fsize is not valid for 1d-conv.')

        return argument

    elif dim == '2d':
        if type(argument) == int:
            return (argument, argument)

        if len(argument) != 2:
            raise RuntimeError('Argument: conv_fsize must be a tuple or list which length is 2 in 2d-conv.')

        return tuple(argument)


class PermuteLayer(nn.Module):
    def __init__(self, dims):
        super(PermuteLayer, self).__init__()

        if not isinstance(dims, (tuple, list)):
            raise TypeError('Argument: dims was recommended to be a tuple.')

        self.dims = tuple(dims)

    def forward(self, input):
        return input.permute(self.dims)

    def __repr__(self):
        return self.__class__.__name__ + '(dims: {0})'.format(self.dims)


class RNNPermuteLayer(PermuteLayer):
    def __init__(self, dims, retain_hidden = False):
        super(RNNPermuteLayer, self).__init__(dims = dims)

        if not isinstance(retain_hidden, bool):
            raise TypeError('Argument: retain_hidden must be a boolean.')

        self.retain_hidden = retain_hidden

    def forward(self, input):
        # index 0 is output of RNN, index 1 is the hidden state of RNN
        input = list(input)

        if self.retain_hidden:
            input[0] = input[0].permute(self.dims)
            return tuple(input)
        else:
            return input[0].permute(self.dims)

    def __repr__(self):
        return self.__class__.__name__ + '(dims: {0}, retain_hidden: {1})'.format(self.dims, self.retain_hidden)


class TorchKernel(nn.Module):
    def __init__(self,
            kernel = 'rbf',
            gamma = 1.0,
            degree = 3,
            coef0 = 0.0,
            support_vectors = None):

        super(TorchKernel, self).__init__()

        if not isinstance(kernel, str):
            raise TypeError('Argument: kernel must be a str.')

        if kernel.lower() not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError('Argument: kernel must be str:linear, str:poly, str:rbf or str:sigmoid.')

        if not isinstance(degree, int):
            raise TypeError('Argument: degree must be a int.')

        if not isinstance(coef0, float):
            raise TypeError('Argument: coef0 must be a float.')

        if not isinstance(gamma, (float, np.ndarray, torch.Tensor)):
            raise TypeError('Argument: gamma must be a float or 0-dim np.ndarray or torch.Tensor.')

        if support_vectors is not None:
            self.support_vectors = self._array_type_check(support_vectors)

        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_vectors = support_vectors

    def forward(self, data, support_vectors = None):
        support_vectors = self._sv_check(support_vectors)
        if self.kernel == 'linear':
            output = self._linear_kernel(data, support_vectors)
        elif self.kernel == 'poly':
            output = self._poly_kernel(data, support_vectors)
        elif self.kernel == 'rbf':
            output = self._rbf_kernel(data, support_vectors)
        elif self.kernel == 'sigmoid':
            output = self._sigmoid_kernel(data, support_vectors)

        return output

    def _linear_kernel(self, data, support_vectors):
        output = torch.matmul(data, support_vectors.permute(1, 0))
        return output

    def _poly_kernel(self, data, support_vectors):
        output = torch.pow((self.gamma * torch.matmul(data, support_vectors.permute(1, 0)) + self.coef0), self.degree)
        return output

    def _sigmoid_kernel(self, data, support_vectors):
        output = torch.tanh(self.gamma * torch.matmul(data, support_vectors.permute(1, 0)) + self.coef0)

    def _rbf_kernel(self, data, support_vectors):
        output = [torch.exp(-self.gamma * torch.pow((support_vectors - data[i]), 2).sum(dim = 1)) for i in range(len(data))]
        output = torch.stack(output, dim = 0)
        return output

    def _sv_check(self, sv):
        if sv is None:
            if self.support_vectors is None:
                raise RuntimeError('Support vectors not set, Please use TorchKernel.replace_sv to input support vectors.')
            else:
                return self.support_vectors

        else:
            return sv


    def _array_type_check(self, array):
        if not isinstance(array, (list, torch.Tensor, np.ndarray)):
            raise TypeError('Input array must be np.ndarray or torch.Tensor.')

        if isinstance(array, list):
            array = torch.tensor(array)

        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)

        if len(array.size()) <= 1:
            raise RuntimeError('Support vectors please input in the shape of (numbers, features).')

        return array

    def add_sv(self, sv):
        self.append_support_vector(sv)
        return None

    def add_support_vector(self, sv):
        self.append_support_vector(sv)

    def append_sv(self, sv):
        self.append_support_vector(sv)
        return None

    def append_support_vector(self, sv):
        sv = self._array_type_check(sv)
        device = self.support_vectors.device
        sv = sv.to(device)

        sv = torch.cat((self.support_vectors, sv), dim = 0)

        return None

    def replace_sv(self, sv):
        sv = self.replace_support_vectors(sv)
        return None

    def replace_support_vectors(self, sv):
        sv = self._array_type_check(sv)
        if self.support_vectors is None:
            device = torch.device('cpu')
        else:
            device = self.support_vectors.device

        self.support_vectors = sv

        return None

    def __repr__(self):
        text = super(TorchKernel, self).__repr__()
        text += '\nOther properties:'
        if self.kernel == 'linear':
            text += '  kernel={0}'.format(self.kernel)
        elif self.kernel == 'poly':
            text += ' kernel={0},\n  gamma={1},\n  coef0={2},\n  degree={3}'.format(self.kernel, self.gamma, self.coef0, self.degree)
        elif self.kernel == 'sigmoid':
            text += ' kernel={0},\n  gamma={1}, \n  coef0={2}'.format(self.kernel, self.gamma, self.coef0)
        elif self.kernel == 'rbf':
            text += '  kernel={0},\n  gamma={1}'.format(self.kernel, self.gamma)

        return text


