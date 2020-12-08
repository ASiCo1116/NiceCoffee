import os

import numpy as np

from .utils import load_pickle_obj
from .base import Report
from .classification_report import ClassificationReport
from .transforms import transform_nir_report 

__all__ = ['Report', 'ClassificationReport', 'load_report', 'transform_nir_report']


def load_report(report_path, best = True):
    if best:
        if os.path.isdir(report_path):
            file_list = os.listdir(report_path)
            best_report_found = False
            for f in file_list:
                if f == 'best_report.report':
                    report_path = os.path.join(report_path, 'best_report.report') 
                    best_report_found = True
        
            if not best_report_found:
                name = os.path.split(report_path)[-1]
                report_path = os.path.join(report_path, name + '.report')

            state = load_pickle_obj(report_path, extension_check = False)

        elif os.path.isfile(report_path):
            state = load_pickle_obj(report_path, extension_check = False)

        else:
            raise TypeError('Argument: report_path must be a string path.')
    else:
        if os.path.isdir(report_path):
            name = os.path.split(report_path)[-1]
            report_path = os.path.join(report_path, name + '.report')

            state = load_pickle_obj(report_path, extension_check = False)

        elif os.path.isfile(report_path):
            state = load_pickle_obj(report_path, extension_check = False)

        else:
            raise TypeError('Argument: report_path must be a string path.')


    if state['type'] == 'ClassificationReport':
        return ClassificationReport(np.array([[]]), np.array([[]])).load(report_path)
    else:
        raise RuntimeError('File cannot be loaded. Please check the report is saved by lib.eval object.')

    return None


