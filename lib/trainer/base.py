import os
import abc
import time
import shutil
import logging
import datetime

from .utils import load_config, init_directory
from ..model import DeepLearningModule, NIRDeepLearningModule

#--------------------------------------------------------------------------------
# trainer.py contains the main process of training ML/DL model
#--------------------------------------------------------------------------------

class BaseTrainer(abc.ABC):
    def __init__(self, config_path, directory_check = True):
        # init I/O
        # directory_check is the backdoor of trainer reusing, not existing argument
        if not isinstance(config_path, str):
            raise TypeError('Argument: config_path must be a string.')

        self.config_path = config_path
        self.config = load_config(config_path)

        if directory_check:
            try:
                self.save_dir = self._init_directory(self.config['model_name'],
                        self.config['overwrite_directory'],
                        self.config['continue_training'])
            except KeyError:
                self.save_dir = self._init_directory(self.config['model_name'],
                        self.config['overwrite_directory'])

    def train(self, config = None):
        # training progress
        if config is None:
            config = self.config

        logging.info('Start training process.')
        start_time = time.time()

        self.model = self._train(config)
        self.record(config)

        total_time = time.time() - start_time
        logging.info('Finish training process. Total time: ' + str(datetime.timedelta(seconds = total_time)))

        return None

    def _train(self, config):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def record(self, config = None):
        if config is None:
            self._record(self.config)
        else:
            self._record(config)

        logging.info('History file saving done.')

        if self.datatype.lower() == 'nir':
            if isinstance(self.model, NIRDeepLearningModule):
                preprocess = self.dataset.catch_transforms(mode = 'both')
            else:
                preprocess = self.dataset.catch_transforms(mode = 'preprocess')

        elif self.datatype.lower() == 'gcms':
            if isinstance(self.model, DeepLearningModule):
                preprocess = self.dataset.catch_transforms(mode = 'both')
            else:
                preprocess = self.dataset.catch_transforms(mode = 'preprocess')

        self.model.set_preprocess(preprocess)
        self.model.save(os.path.join(self.save_dir, config['model_name']))
        logging.info('Final trained model saving done.')

        return None

    def _record(self, config):
        raise NotImplementedError()

    def _init_directory(self,
            model_name,
            overwrite = False,
            continue_training = False):

        # information saving directory
        # save model checkpoint and episode history

        init_directory('outputs')
        save_dir = os.path.join('outputs', model_name)
        init_directory(save_dir)

        if overwrite:
            logging.warning('All data in the directory will be overwrite.')
            shutil.rmtree(save_dir, ignore_errors = True)

        self.continue_training = continue_training
        if not overwrite and len(os.listdir(save_dir)) != 0:
            raise RuntimeError('Files exist in the directory, please use overwrite_directory = True or rename directory.')

        shutil.copy(self.config_path, os.path.join(save_dir, os.path.split(self.config_path)[-1]))
        logging.info('All object (model checkpoint, trainning history, ...) would save in ' + save_dir)

        return save_dir


