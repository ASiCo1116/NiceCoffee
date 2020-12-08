import logging

import numpy as np

from .classification_report import ClassificationReport
from ..data.utils import select_nir_label_list, flavor_label_transform_matrix

#--------------------------------------------------------------------------------
# transforms.py contains some reprot transform function and class
#--------------------------------------------------------------------------------


__all__ = ['transform_nir_report']


def transform_nir_report(report, old_label, new_label):
    if not isinstance(report, ClassificationReport):
        raise TypeError('Argument: report must be a ClassificationReport object.')

    old_label_list = select_nir_label_list(old_label)
    old_label_list = old_label_list[: -1]
    new_label_list = select_nir_label_list(new_label)
    new_label_list = new_label_list[: -1]

    if ('baked' in old_label_list) and ('baked' not in new_label_list):
        raise RuntimeError('The label with baked need to transform into a label with baked.')

    if ('baked' not in old_label_list) and ('baked' in new_label_list):
        raise RuntimeError('The new label contained baked label, but the old label do not contain this category.')

    if len(new_label_list) > len(old_label_list):
        raise RuntimeError('The function only support complicated label transform into simple label.')

    transform_matrix = flavor_label_transform_matrix(old_label, new_label)
    old_prediction, old_target = report.prediction, report.label
    new_prediction = np.matmul(old_prediction.astype(np.float), transform_matrix.astype(np.float))
    new_target = np.matmul(old_target.astype(np.float), transform_matrix.astype(np.float))

    new_prediction = np.clip(new_prediction, 0., 1.)
    new_target = np.clip(new_target, 0., 1.)

    new_report = ClassificationReport(new_prediction.astype(np.int),
            new_target.astype(np.int),
            new_label_list)

    return new_report


