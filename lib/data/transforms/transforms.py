from __future__ import division
from __future__ import print_function

import math
import logging
import numbers
import collections
import random
import numpy as np

import torch
import torch.nn.functional as F

from .base import TransformFunctionObject

#--------------------------------------------------------------------------------
# transforms.py contains all augmentation function for mass spectrum or spectrum
# data.
# Please use lib.data.transforms.Compose to encapsulate all needed function.
#-------------------------------------------------------------------------------- 


__all__ = ['Compose', 'ToTensor', 'ToNumpy', 'Scale', 'Standardize', 'SNV', 'Pad', 'Flatten',
        'Shift', 'RandomShift', 'Noise', 'AddedGaussianNoise', 'MultipliedGaussianNoise',
        'AddedBetaNoise','MultipliedBetaNoise', 'AddedFNoise', 'MultipliedFNoise',
        'AddedStudentTNoise', 'MultipliedStudentTNoise', 'AddedUniformNoise', 
        'MultipliedUniformNoise', 'AddedWeibullNoise', 'MultipliedWeibullNoise']


def _tensor_check(obj):
    if not isinstance(obj, torch.Tensor):
        raise TypeError('Input object must be a torch.Tensor.')

    return None

def _random_apply(spectrum, func, p = 0.5):
    if random.random() < 0.5:
        return spectrum
    else:
        return func(spectrum)


class Compose(TransformFunctionObject):
    def __init__(self, transforms):
        if not isinstance(transforms, (list, tuple)):
            raise TypeError('Input function must be in list or tuple.')

        for func in transforms:
            if not isinstance(func, TransformFunctionObject):
                raise TypeError('Input function in list must in the lib.data.transform, or please self-define it.')

        self.transforms = transforms

    def __call__(self, spectrum):
        for t in self.transforms:
            spectrum = t(spectrum)

        return spectrum

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)

        format_string += '\n)'
        return format_string


class ToTensor(TransformFunctionObject):
    # Convert spectrum data (numpy.ndarray or list)to torch.Tensor
    def __call__(self, spectrum):
        if not isinstance(spectrum, (list, np.ndarray)):
            raise TypeError('Only numpy.ndarray or list data can be convert into tensor.')

        return torch.tensor(spectrum)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpy(TransformFunctionObject):
    # Convert tensor spectrum data to numpy.ndarray
    def __call__(self, tensor):
        _tensor_check(tensor)
        return tensor.numpy()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Scale(TransformFunctionObject):
    # Scale spectrum from min value to the max value in specific dimension
    def __init__(self, min = 0, max = 1, inplace = False):
        self.min = min
        self.max = max
        self.inplace = inplace

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        if not self.inplace:
            spectrum = spectrum.clone()

        dtype = spectrum.dtype
        min = torch.as_tensor(self.min, dtype = dtype, device = torch.device)
        max = torch.as_tensor(self.max, dtype = dtype, device = torch.device)
        max_value = spectrum.max()
        min_value = spectrum.min()
        scale_factor = (max_value - min_value) / (max - min)
        spectrum = (spectrum - torch.full_like(spectrum, min_value)) * scale_factor
        return spectrum.detach()

    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1}, inplace={2})'.format(self.min, self.max, self.inplace)


class Flatten(TransformFunctionObject):
    # Flatten the mass spectrum to n-d array
    def __init__(self, inplace = False):
        self.inplace = inplace

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        if not self.inplace:
            spectrum = spectrum.clone()

        return spectrum.reshape(-1).detach()

    def __repr__(self):
         return self.__class__.__name__ + '(inplace={0})'.format(self.inplace)
            

class Normalize(TransformFunctionObject):
    # Normalize the spectrum by the mean and standard deviation
    def __init__(self, mean = 0., std = 1., inplace = False):
        self.mean = mean
        if std == 0.0:
            raise ValueError('Standard deviaiton can not be zero')

        self.std = std
        self.inplace = inplace

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        if not self.inplace:
            spectrum = spectrum.clone()

        dtype = spectrum.dtype
        mean = torch.as_tensor(mean, dtype = dtype, device = torch.device)
        std = torch.as_tensor(std, dtype = dtype, device = torch.device)
        spectrum = (spectrum - torch.full_like(spectrum, mean)) / torch.full_like(spectrum, std)
        return spectrum.detach()

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(self.mean, self.std, self.inplace)


class Standardize(TransformFunctionObject):
    # Standardize a sepctrum in specfic dimension
    # SNV is also implemetned by same class
    def __init__(self, dim = 0, inplace = False):
        self.dim = dim
        self.inplace = inplace

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        if not self.inplace:
            spectrum = spectrum.clone()

        means = spectrum.mean(dim = self.dim, keepdim = True)
        stds = spectrum.std(dim = self.dim, keepdim = True)
        spectrum = (spectrum - means) / stds
        return spectrum.detach()

    def __repr__(self):
        return self.__class__.__name__ + '(dim={0}, inplace={1})'.format(self.dim, self.inplace)


class SNV(Standardize):
    # standard normal variate
    def __init__(self, inplace = False):
        super(SNV, self).__init__(dim = 0, inplace = inplace)

    def _size_check(self, spectrum):
        if len(spectrum.size()) != 1:
            raise RuntimeError('SNV only can be used on one dimensional spectrum tensor.')

        return spectrum

    def __call__(self, spectrum):
        if not self.inplace:
            spectrum = spectrum.clone()

        spectrum = self._size_check(spectrum)
        spectrum = super(SNV, self).__call__(spectrum)
        return spectrum

    def __repr__(self):
        return self.__class__.__name__ + '(inplace={0})'.format(self.inplace)

class Pad(TransformFunctionObject):
    # pad spectrum data on left and right side (axis=0)
    def __init__(self, padding, fill = 0, padding_mode = 'constant'):
        if not isinstance(padding, (numbers.Number, tuple)):
            raise ValueError('Argument: padding must be a number or tuple.')

        self.padding = padding
        if not isinstance(fill, (numbers.Number, tuple)):
            raise ValueError('Argument: fill must be a number or tuple.')

        self.fill = fill
        if padding_mode not in ['constant', 'edge']:
            raise ValueError('Argument: padding mode must in [constant, edge]')

        if self.padding_mode == 'constant':
            if len(spectrum.size()) != 1:
                logging.warning('2d or upper dimensional spectrum might lose lots of ' +
                    'information, if it pad with constant.')

        self.padding_mode = padding_mode
        if isinstance(padding, collections.Sequence) and len(padding) != 2:
            raise ValueError("Padding must be an int or a 2 element tuple," + 
                " not a {} element tuple".format(len(padding)))

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        shape = spectrum.size()
        if self.padding_mode == 'constant':
            if isinstance(padding, collections.Sequence):
                shape[0] = shape[0] + padding[0] + padding[1]
                new = torch.empty(shape)
                new[0: padding[0]] = self.fill
                new[padding[0]: (padding[0] + spectrum.size()[0])] = spectrum
                new[(padding[0] + spectrum.size()[0]):] = self.fill
                return new.detach()
            else:
                shape[0] += (padding * 2)
                new = torch.empty(shape)
                new[0: padding] = self.fill
                new[padding: (padding + spectrum.size()[0])] = spectrum
                new[(padding + spectrum.size()[0]):] = self.fill
                return new.detach()
        elif self.padding_mode == 'edge':
            if self.fill != 0:
                logging.warning('Argument fill will not used in edge padding mode.')

            if isinstance(padding, collections.Sequence):
                shape[0] = shape[0] + padding[0] + padding[1]
                new = torch.empty(shape)
                new[0: padding[0], :] = spectrum[0]
                new[padding[0]: (padding[0] + spectrum.size()[0])] = spectrum
                new[(padding[0] + spectrum.size()[0]):] = spectrum[-1]
                return new.detach()
            else:
                shape[0] += (padding * 2)
                new = torch.empty(shape)
                new[0: padding] = spectrum[0]
                new[padding: (padding + spectrum.size()[0])] = spectrum
                new[(padding + spectrum.size()[0]):] = spectrum[-1]
                return new.detach()

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Shift(Pad):
    # shift spectrum to lefter side or righter side
    def __init__(self, length, direction = 'random', compensation = 'edge', fill = 0, inplace = False):
        if not isinstance(length, (int, tuple)):
            raise TypeError('Shift length must feed in int or tuple with min length to max length.')

        if isinstance(length, tuple):
            logging.warning('Shift length will be random decided in Shift if length is a tuple.')

        self.length = length
        if direction not in ['left', 'right', 'random']:
            raise ValueError('Argument: direction must be in [left, right, random].')

        self.direction = direction
        if compensate not in ['edge', 'constant']:
            raise ValueError('Argument: compensate must in [constant, edge].')

        if compensation == 'constant':
            logging.warning('if argument: compensation is constant, spectrum will fill in a value,' + 
                    'and in most of condition, filling in constant might not be a suitable choice.')

        self.compensation = compensation
        self.fill = fill
        self.inplace = inplace
        super(Shift, self).__init__(length, fill, inplace)

    def __call__(self, spectrum):
        length = spectrum.size(0)
        if self.direction == 'random':
            if random.random() < 0.5:
                direction = 'left'
            else:
                direction = 'right'
        else:
            direction = self.direction

        spectrum = super(Shift, self).__call__(spectrum)
        if direction == 'left':
            spectrum = spectrum[self.length * 2:]
        elif direction == 'right':
            spectrum = spectrum[0: length]

        return spectrum

    def __repr__(self):
        if self.compensate == 'edge':
            return self.__class__.__name__ + '(length={0}, direction={1}, compensate={2}, inplace={3})'.\
                format(self.length, self.direction, self.compensation, self.inplace)

        elif self.compensate == 'constant':
            return self.__class__.__name__ + '(length={0}, direction={1}, compensate={2}, fill={3}, inplace={4})'.\
                format(self.length, self.direction, self.compensation, self.fill, self.inplace)


class RandomShift(Shift):
    def __init__(self, length, p = 0.5, direction = 'random', compensation = 'edge', fill = 0, inplace = False):
        if not isinstance(length, (int, tuple)):
            raise TypeError('Shift length must feed in int or tuple with min length to max length.')

        if isinstance(length, int) and length >= 0:
            self.length = (0, length)
        elif isinstance(length, tuple):
            self.length = length
        else:
            raise ValueError('Shift length must be a positive int.')

        self.p = p
        super(RandomShift, self).__init__(length = self.length,\
            direction = direction, compensation = compensation, fill = fill, inplace = inplace)

    def __call__(self, spectrum):
        return _random_apply(spectrum, super(RandomShift, self).__call__, self.p)

    def __repr__(self):
        if self.compensate == 'edge':
            return self.__class__.__name__ + '(length={0}, p={1}, direction={2}, compensate={3}, inplace={4})'.\
                format(self.length, self.p, self.direction, self.compensation, self.inplace)

        elif self.compensate == 'constant':
            return self.__class__.__name__ + '(length={0}, p={1}, direction={2}, compensate={3}, fill={4}, inplace={5})'.\
                format(self.length, self.p, self.direction, self.compensation, self.fill, self.inplace)


class Noise(TransformFunctionObject):
    # base on numpy.random
    # beta, noraml, f, student-t, uniform, weibull
    def __init__(self, distribution, dim, operator, inplace = False, *args, **kwargs):
        self.distribution = distribution
        self.dist_func = self._select_distribution(distribution)
        if not isinstance(dim, int):
            raise TypeError('Argument: dim must be an int.')

        self.dim = dim
        if operator not in ['add', 'multiply']:
            raise ValueError('Argument: operator must be str(add) or str(multiply).')

        self.operator = operator
        self.inplace = inplace
        self.args = args
        self.kwargs = kwargs

    def _select_distribution(self, select):
        if select.lower() == 'beta':
            return np.random.beta
        elif select.lower() == 'f':
            return np.random.f
        elif select.lower() == 'noraml' or select.lower() == 'gaussian':
            return np.random.normal
        elif select.lower() == 'student-t' or select.lower() == 'student_t':
            return np.random.standard_t
        elif select.lower() == 'uniform':
            return np.random.uniform
        elif select.lower() == 'weibull':
            return np.random.weibull
        else:
            raise ValueError(select, 'is invalid for distributions selection.')

    def __call__(self, spectrum):
        _tensor_check(spectrum)
        if not self.inplace:
            specturm = spectrum.clone()

        repeat_size = list(spectrum.size())
        repeat_size[self.dim] = 1
        noise = self.dist_func(size = spectrum.size(self.dim), *self.args, **self.kwargs)
        noise = torch.as_tensor(noise, dtype = spectrum.dtype, device = spectrum.device)
        noise = noise.repeat(repeat_size).view(-1).view(spectrum.size())
        if self.operator == 'add':
            spectrum += noise
        elif self.operator == 'multiply':
            spectrum *= noise

        return spectrum.detach()

    def __repr__(self):
        return self.__class__.__name__ + '(distribution={0}, dim={1}, args={2}, kwargs={3})'.\
            format(self.distribution, self.dim, self.args, self.kwargs)


class AddedGaussianNoise(Noise):
    def __init__(self, mean = 0., std = 0.001, dim = 0, inplace = False):
        if not isinstance(mean, numbers.Number):
            raise TypeError('Argument: mean must be a number representing the mean of Gaussian distribution.')

        self.mean = mean
        if not isinstance(std, numbers.Number):
            raise TypeError('Argument: std must be a number representing the std of Gaussian distribution.')

        self.std = std
        super(AddedGaussianNoise, self).__init__('gaussian', dim, 'add', inplace, loc = mean, scale = std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, dim={2}, inplace={3})'.\
            format(self.mean, self.std, self.dim, self.inplace)


class MultipliedGaussianNoise(Noise):
    def __init__(self, mean = 1., std = 0.001, dim = 0, inplace = False):
        if not isinstance(mean, numbers.Number):
            raise TypeError('Argument: mean must be a number representing the mean of Gaussian distribution.')

        self.mean = mean
        if not isinstance(std, numbers.Number):
            raise TypeError('Argument: std must be a number representing the std of Gaussian distribution.')

        self.std = std
        super(MultipliedGaussianNoise, self).__init__('gaussian', dim, 'multiply', inplace, loc = mean, scale = std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, dim={2}, inplace={3})'.\
            format(self.mean, self.std, self.dim, self.inplace)


class AddedBetaNoise(Noise):
    def __init__(self, a, b, dim = 0, inplace = False):
        if not isinstance(a, numbers.Number):
            raise TypeError('Argument: a must be a number representing alpha of Beta distribution.')

        self.a = a
        if not isinstance(b, numbers.Number):
            raise TypeError('Argument: b must be a number representing beta of Beta distribution.')

        self.b = b
        super(AddedBetaNoise, self).__init__('beta', dim, 'add', inplace, a = a, b = b)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1}, dim={2}, inplace={3})'.\
            format(self.a, self.b, self.dim, self.inplace)


class MultipliedBetaNoise(Noise):
    def __init__(self, a, b, dim = 0, inplace = False):
        if not isinstance(a, numbers.Number):
            raise TypeError('Argument: a must be a number representing alpha of Beta distribution.')

        self.a = a
        if not isinstance(b, numbers.Number):
            raise TypeError('Argument: b must be a number representing beta of Beta distribution.')

        self.b = b
        super(MultipliedBetaNoise, self).__init__('beta', dim, 'multiply', inplace, a = a, b = b)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1}, dim={2}, inplace={3})'.\
            format(self.a, self.b, self.dim, self.inplace)


class AddedFNoise(Noise):
    def __init__(self, dfnum, dfden, dim = 0, inplace = False):
        if not isinstance(dfnum, numbers.Number):
            raise TypeError('Argument: dfnum must be a number representing d1 of F distribution.')

        self.dfnum = dfnum
        if not isinstance(dfden, numbers.Number):
            raise TypeError('Argument: dfden must be a number representing d2 of F distribution.')

        self.dfden = dfden
        super(AddedFNoise, self).__init__('f', dim, 'add', inplace, dfnum = dfnum, dfden = dfden)

    def __repr__(self):
        return self.__class__.__name__ + '(dfnum={0}, dfden={1}, dim={2}, inplace={3})'.\
            format(self.dfnum, self.dfden, self.dim, self.inplace)


class MultipliedFNoise(Noise):
    def __init__(self, dfnum, dfden, dim = 0, inplace = False):
        if not isinstance(dfnum, numbers.Number):
            raise TypeError('Argument: dfnum must be a number representing d1 of F distribution.')

        self.dfnum = dfnum
        if not isinstance(dfden, numbers.Number):
            raise TypeError('Argument: dfden must be a number representing d2 of F distribution.')

        self.dfden = dfden
        super(MultipliedFNoise, self).__init__('f', dim, 'multiply', inplace, dfnum = dfnum, dfden = dfden)

    def __repr__(self):
        return self.__class__.__name__ + '(dfnum={0}, dfden={1}, dim={2}, inplace={3})'.\
            format(self.dfnum, self.dfden, self.dim, self.inplace)


class AddedStudentTNoise(Noise):
    def __init__(self, df, dim = 0, inplace = False):
        if not isinstance(df, numbers.Number):
            raise ValueError('Argument: df must be a number representing df of student-t distribution.')

        self.df = df
        super(AddedStudentTNoise, self).__init__('student_t', dim, 'add', inplace, df = df)

    def __repr__(self):
        return self.__class__.__name__ + '(df={0}, dim={1}, inplace={2})'.\
            format(self.df, self.dim, self.inplace)


class MultipliedStudentTNoise(Noise):
    def __init__(self, df, dim = 0, inplace = False):
        if not isinstance(df, numbers.Number):
           raise ValueError('Argument: df must be a number representing df of student-t distribution.')

        self.df = df
        super(MultipliedStudentTNoise, self).__init__('student_t', dim, 'multiply', inplace, df = df)

    def __repr__(self):
        return self.__class__.__name__ + '(df={0}, dim={1}, inplace={2})'.\
            format(self.df, self.dim, self.inplace)


class AddedUniformNoise(Noise):
    def __init__(self, low, high, dim = 0, inplace = True):
        if not isinstance(low, numbers.Number):
            raise ValueError('Argument: low must be a number representing minimum of uniform distribution.')

        self.low = low
        if not isinstance(high, numbers.Number):
            raise ValueError('Argument: high must be a number representing maximum of uniform distribution.')

        self.high = high
        super(AddedUniformNoise, self).__init__('uniform', dim, 'add', inplace, low = low, high = high)

    def __repr__(self):
        return self.__class__.__name__ + '(low={0}, high={1}, dim={2}, inplace={3})'.\
            format(self.low, self.high, self.dim, self.inplace)


class MultipliedUniformNoise(Noise):
    def __init__(self, low, high, dim = 0, inplace = True):
        if not isinstance(low, numbers.Number):
            raise ValueError('Argument: low must be a number representing minimum of uniform distribution.')

        self.low = low
        if not isinstance(high, numbers.Number):
            raise ValueError('Argument: high must be a number representing maximum of uniform distribution.')

        self.high = high
        super(MultipliedUniformNoise, self).__init__('uniform', dim, 'multiply', inplace, low = low, high = high)

    def __repr__(self):
        return self.__class__.__name__ + '(low={0}, high={1}, dim={2}, inplace={3})'.\
            format(self.low, self.high, self.dim, self.inplace)


class AddedWeibullNoise(Noise):
    def __init__(self, a, dim = 0, inplace = False):
        if not isinstance(a, numbers.Number):
            raise ValueError('Argument: a must be a number representing a of weibull distribution.')

        self.a = a
        super(AddedWeibullNoise, self).__init__('weibull', dim, 'add', inplace, a = a)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, dim={1}, inplace={2})'.\
            format(self.a, self.dim, self.inplace)

class MultipliedWeibullNoise(Noise):
    def __init__(self, a, dim = 0, inplace = False):
        if not isinstance(a, numbers.Number):
            raise ValueError('Argument: a must be a number representing a of weibull distribution.')

        self.a = a 
        super(MultipliedWeibullNoise, self).__init__('weibull', dim, 'multiply', inplace, a = a)

    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, dim={1}, inplace={2})'.\
            format(self.a, self.dim, self.inplace)


