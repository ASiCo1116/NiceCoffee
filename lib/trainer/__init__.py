from .base import BaseTrainer
from .machine_learning import MachineLearningTrainer
from .deep_learning import DeepLearningTrainer, DeepLearningClassifierTrainer
from .utils import load_config


__all__ = ['BaseTrainer', 'MachineLearningTrainer', 'DeepLearningTrainer',
        'DeepLearningClassifierTrainer', 'select_trainer']


def select_trainer(config_path, directory_check = True):
    config = load_config(config_path)

    try:
        if config['trainer'].lower() == 'nir_ml_classifier':
            return MachineLearningTrainer(config_path, datatype = 'nir', directory_check = directory_check)
        elif config['trainer'].lower() == 'gcms_ml_classifier':
            return MachineLearningTrainer(config_path, datatype = 'gcms', directory_check = directory_check)
        elif config['trainer'].lower() == 'nir_dl_classifier':
            return DeepLearningClassifierTrainer(config_path, datatype = 'nir', directory_check = directory_check) 
        elif config['trainer'].lower() == 'gcms_dl_classifier':
            return DeepLearningClassifierTrainer(config_path, datatype = 'gcms', directory_check = directory_check)
        else:
            raise ValueError('Argument: trainer must is not valid, please check configs/*.yaml')

    except KeyError:
        raise RuntimeError('Config file does not contain trainer information, please check the document.')


