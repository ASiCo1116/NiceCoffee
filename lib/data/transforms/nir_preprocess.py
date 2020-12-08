import logging

import numpy as np

from scipy.signal import savgol_filter

from .base import TransformFunctionObject
from ...utils import transform_empty_argument

#--------------------------------------------------------------------------------
# the nir_preprocess.py is used in the flavor model of the coffee flavor recognition
# the nir_preprocess.py contain all the function used in the project of coffee NIR
#--------------------------------------------------------------------------------


__all__ = ['NIRMSC', 'NIRSNV', 'NIRSG', 'NIRPreprocess']


class NIRPreprocess(TransformFunctionObject):
    def __init__(self, preprocess):
        if not isinstance(preprocess, (list, tuple)):
            raise TypeError('Argument: preprocess must be a list or tuple.')

        self.preprocess = preprocess
        self.preprocess_func = self._init_preprocess(preprocess)

    def __call__(self, spectrum, train = False):
        for p in self.preprocess_func:
            spectrum = p(spectrum, train = train)

        return spectrum

    def _init_preprocess(self, preprocess_list):
        func_list = []
        for f in preprocess_list:
            argument = transform_empty_argument(f, 'argument')
            if f['name'].lower() == 'msc':
                func_list.append(NIRMSC(**argument))
            elif f['name'].lower() == 'snv':
                func_list.append(NIRSNV(**argument))
            elif f['name'].lower() == 'sg':
                func_list.append(NIRSG(**argument))
            else:
                 raise NotImplementedError(f['name'], ' is not availalbe in this program.')

        return func_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess_func:
            format_string += '\n'
            format_string += '    {0}'.format(t)

        format_string += '\n)'
        format_string += '\nThis NIRPreprocess object was inherited from old project, transform function is recommended.'
        return format_string


class NIRMSC(TransformFunctionObject):
    def __init__(self):
        self.reference = None
        self.ref_exist = False

    def __call__(self, spectrum, train = False):
        #the function is the correction of the NIR data
        #input data is a samples * one dimension  feature data
    
        #mean central correction
        for i in range(spectrum.shape[0]):
            spectrum[i, :] -= spectrum[i, :].mean()
    
        if not train and self.reference is not None:
            ref = self.reference
        else:
            ref = np.mean(spectrum, axis = 0)
            self.reference = ref
    
        #define a new array for the data after correction
        msc_spectrum = np.zeros_like(spectrum)
        for i in range(spectrum.shape[0]):
            #to run the regression for correction
            fit = np.polyfit(ref, spectrum[i, :], 1, full = True)
            #apply correction
            msc_spectrum[i, :] = (spectrum[i, :] - fit[0][1]) / fit[0][0]
    
        if train:
            self.reference = ref
            self.ref_exist = True
    
        return msc_spectrum

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NIRSNV(TransformFunctionObject):
    def __call__(self, spectrum, train = False):
        #the function is the correction of the NIR data
        #input data is a samples * one dimension  feature data
    
        #define a new array for the data after correction
        snv_spectrum = np.zeros_like(spectrum)
        for i in range(spectrum.shape[0]):
            snv_spectrum[i, :] = (spectrum[i, :] - np.mean(spectrum[i, :])) / np.std(spectrum[i, :])
    
        return snv_spectrum

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NIRSG(TransformFunctionObject):
    def __init__(self, window_length, poly_order, derivative_order):
       if not isinstance(window_length, int):
           raise TypeError('Argument: window_length must be a int.')

       if window_length % 2 == 0 or window_length < 1:
           raise ValueError('Argument: window_length of SG filter must be a positive and odd number.')

       if not isinstance(poly_order, int):
           raise TypeError('Argument: poly_order must be a int.')

       if poly_order < 0:
           raise ValueError('Argument: poly_order of SG filter must be a positive.')

       if not isinstance(derivative_order, int):
           raise TypeError('Argument: derivative_order must be a int.')

       if poly_order < 0:
           raise ValueError('Argument: derivative_order of SG filter must be a positive.')

       self.window_length = window_length
       self.poly_order = poly_order
       self.derivative_order = derivative_order

    def __call__(self, spectrum, train = False):
        for i in range(spectrum.shape[0]):
            spectrum[i] = savgol_filter(spectrum[i],
                    window_length =  self.window_length,
                    polyorder = self.poly_order,
                    deriv = self.derivative_order)
    
        if self.window_length == 1:
            adjust_length = 0
        else:
            adjust_length = self.window_length // 2
    
        if adjust_length > 0:
            spectrum[:, -adjust_length:] = 0.0
            spectrum[:, 0: adjust_length] = 0.0
    
        return spectrum

    def __repr__(self):
        return self.__class__.__name__ + '(window_lengt={0}, poly_order={1}, derivative_order={2})'.\
            format(self.window_length, self.poly_order, self.derivative_order)


