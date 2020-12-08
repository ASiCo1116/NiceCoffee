import random

import numpy as np

import torch

#--------------------------------------------------------------------------------
# utils.py contains the function which would be used in dataset object
#--------------------------------------------------------------------------------


__all__ = ['transform_labels', 'random_initial_seed']


# transform the original descriptions into self-defined categories in flavor wheel
def transform_labels(layer, wheel, labels, label_descriptions):
    transform_matrix, wheel_descriptions = wheel.label_transform_matrix(label_descriptions, layer)
    labels = np.matmul(labels, transform_matrix)
    labels[labels > 1] = 1
    return labels, wheel_descriptions

def random_initial_seed(min = 1, max = 32767):
    seed = random.randint(min, max)
    return seed


