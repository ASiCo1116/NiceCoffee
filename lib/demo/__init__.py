from .base import BaseDemoObject
from .classifier import ClassifierDemo
from .utils import load_config


__all__ = ['select_demo', 'BaseDemoObject', 'ClassifierDemo']


def select_demo(config_path):
    config = load_config(config_path)

    try:
        if config['demo_task'].lower() == 'nir_flavor_classifier':
            return ClassifierDemo(config_path, datatype = 'nir')
        elif config['demo_task'].lower() == 'gcms_odor_classifier':
            return ClassifierDemo(config_path, datatype = 'gcms')
        else:
            raise ValueError('Argument: demo_task is not valid, please check configs/*.yaml')

    except KeyError:
        raise RuntimeError('Config file does not contain demo_task information, please check the document.')


