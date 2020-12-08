import pickle
import logging

import numpy as np

#--------------------------------------------------------------------------------
# utils.py contains the function used in eval methods.
#--------------------------------------------------------------------------------


__all__ = ['numpy_instance_check', 'load_pickle_obj']


def numpy_instance_check(array, dim = None):
    if not isinstance(array, np.ndarray):
        raise TypeError('Input array must be a numpy.ndarray.')

    if dim is not None:
        if not isinstance(dim, int):
            raise TypeError('Argument: dim in numpy_instance_check must be a int.')

        if array.ndim != dim:
            raise RuntimeError('The dimension of input array must be ' + str(dim))

    return array

def load_pickle_obj(fname, extension_check = True):
    # the function is used to read the data in saved in pickle format
    if extension_check:
        if not fname.endswith('.pkl'):
            raise RuntimeError(fname, 'is not a pickle file.')

    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

    return None


