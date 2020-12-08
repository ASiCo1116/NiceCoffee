import logging

from .transforms import *
from .combined_transforms import CombinedTransforms

#--------------------------------------------------------------------------------
# utils.py contains the str select function for transform used in yaml config.
#--------------------------------------------------------------------------------


__all__ = ['select_transforms', 'combine_transforms_and_preprocess']


def select_transforms(str_list):
    if not isinstance(str_list, (list, tuple)):
        raise TypeError('Input datatype must be a list or tuple.')

    func_list = []
    for f in str_list:
        if f.lower() == 'totensor':
            func_list.append(ToTensor())
        elif f.lower() == 'tonumpy':
            func_list.append(ToNumpy())
        elif f.lower() == 'scale':
            func_list.append(Scale())
        elif f.lower() == 'standardize':
            func_list.append(Standardize())
        elif f.lower() == 'pad':
            func_list.append(Pad())
        elif f.lower() == 'flatten':
            func_list.append(Flatten())
        elif f.lower() == 'shift':
            func_list.append(Shift())
        elif f.lower() == 'randomshift':
            func_list.append(RandomShift())
        elif f.lower() == 'noise':
            func_list.append(Noise())
        elif f.lower() == 'addedgaussiannoise':
            func_list.append(AddedGaussianNoise())
        elif f.lower() == 'multipliedgaussiannoise':
            func_list.append(MultipliedGaussianNoise())
        elif f.lower() == 'addedbetanoise':
            func_list.append(AddedBetaNoise())
        elif f.lower() == 'multipliedbetanoise':
            func_list.append(MultipliedBetaNoise())
        elif f.lower() == 'addedfnoise':
            func_list.append(AddedFNoise())
        elif f.lower() == 'multipliedfnoise':
            func_list.append(MultipliedFNoise())
        elif f.lower() == 'addedstudenttnoise':
            func_list.append(AddedStudentTNoise())
        elif f.lower() == 'multipliedstudenttnoise':
            func_list.append(MultipliedStudentTNoise())
        elif f.lower() == 'addeduniformnoise':
            func_list.append(AddedUniformNoise())
        elif f.lower() == 'multiplieduniformnoise':
            func_list.append(MultipliedUniformNoise())
        elif f.lower() == 'addedweibullnoise':
            func_list.append(AddedWeibullNoise())
        elif f.lower() == 'multipliedweibullnoise':
            func_list.append(MultipliedWeibullNoise())
        else:
            logging.warning(f + ' is not valid transform funciton, please check lib.data.transforms')

    compose = Compose(func_list)

    return compose

def combine_transforms_and_preprocess(transforms = None, preprocess = None):
    return CombinedTransforms(transforms = transforms, preprocess = preprocess)


