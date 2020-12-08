import os
import math
import random
import logging

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data import GCMSClassifierDataset, NIRClassifierDataset
from ..eval import ClassificationReport
from ..model import model_from_config, loss_select
from ..utils import transform_empty_argument 

from ..model import DeepClassifier
from .utils import init_torch_device, init_directory, optim_select, Checkpoint
from .base import BaseTrainer

#--------------------------------------------------------------------------------
# deep_learning.py contains the extended object of lib.trainer.base.BaseTrainer
# the object is used in deep learning model training
#--------------------------------------------------------------------------------


__all__ = ['DeepLearningTrainer', 'DeepLearningClassifierTrainer']


class DeepLearningTrainer(BaseTrainer):
    def __init__(self,
            config_path,
            datatype = 'gcms',
            directory_check = True):

        super(DeepLearningTrainer, self).__init__(
                config_path = config_path,
                directory_check = directory_check)

        if not isinstance(datatype, str):
            raise TypeError('Argument: datatype must be a string.')
        
        if datatype.lower() not in ['nir', 'gcms']:
            raise ValueError('Argument: datatype in lib.trainer.MachineLearningTrainer must be NIR or GCMS.')

        self.datatype = datatype

        self.device = self._init_device(self.config['device'],
                multi_gpu = self.config['multi-gpu'])

        self.info_iter = 100
        try:
            # if the checkpoint is used, would not raise KeyError
            self.checkpoint = Checkpoint(save_dir = self.save_dir,
                    **self.config['checkpoint'])

        except KeyError:
            self.checkpoint = None

        try:
            self.gradient_clipping = self.config['gradient_clipping']

            if self.gradient_clipping['method'].lower() not in ['norm', 'value']:
                raise ValueError("Argument: method in gradient clipping must be 'norm' or 'value'.")

            if not isinstance(self.gradient_clipping['value'], (int, float)):
                raise TypeError('Argument: value in gradient clipping must be positive int or float.')

            if self.gradient_clipping['value'] <= 0:
                raise ValueError('Argument: value in gradient clipping must be positive int or float.')

        except KeyError:
            self.gradient_clipping = None

        self.writer = SummaryWriter(log_dir = self.save_dir,
                comment = self.config['model_name'])
        
    def _train(self, config = None):
        if config is None:
            config = self.config

        dataset = self._init_dataset(config)

        model = self._init_model(config)
        model.train_mode = True
        model = model.float().to(self.device)
        model = self._tune_model(model, config, dataset)

        return model

    def _init_dataset(self, config = None):
        if config is None:
            config = self.config

        if self.datatype == 'gcms':
            dataset = GCMSClassifierDataset(**config['dataset'])
        elif self.datatype == 'nir':
            dataset = NIRClassifierDataset(**config['dataset'])

        self.dataset = dataset

        return dataset

    def _init_model(self, config = None):
        if config is None:
            config = self.config

        model_object = model_from_config(config)
        if self.continue_training:
            model = model_from_config(self.save_dir)
        else:
            model_config = transform_empty_argument(config, 'argument')
            model = model_object(input_size = self.dataset.data_size(),
                    output_size = self.dataset.descriptions(),
                    **model_config)

        logging.info('Model information:')
        model.summary()

        return model

    def _init_device(self, device, multi_gpu = False):
        device = init_torch_device(device, multi_gpu = multi_gpu)
        return device


class DeepLearningClassifierTrainer(DeepLearningTrainer):
    def __init__(self,
            config_path,
            datatype = 'gcms',
            directory_check = True):

        super(DeepLearningClassifierTrainer, self).__init__(
                config_path = config_path,
                datatype = datatype,
                directory_check = directory_check)

    def eval(self, model, dataset):
        with torch.no_grad():
            dataset.set_mode('train')
            dataloader = DataLoader(dataset, batch_size = self.batch_size)
            self.model, optim, _iter_index, _train_finish, train_report = self._loop(
                    0,
                    float('inf'),
                    self.model,
                    None,
                    self.loss_func,
                    dataloader,
                    'test')

            dataset.set_mode('val')
            dataloader = DataLoader(dataset, batch_size = self.batch_size)
            self.model, optim, _iter_index, _train_finish, val_report = self._loop(
                    0,
                    float('inf'),
                    self.model,
                    None,
                    self.loss_func,
                    dataloader,
                    'test')

            dataset.set_mode('test')
            dataloader = DataLoader(dataset, batch_size = self.batch_size)
            self.model, optim, _iter_index, _train_finish, test_report = self._loop(
                    0,
                    float('inf'),
                    self.model,
                    None,
                    self.loss_func,
                    dataloader,
                    'test')

        return train_report, val_report, test_report 

    def _record(self, config):

        train_report, val_report, test_report = self.eval(self.model, self.dataset)

        logging.info('\nTraining set:')
        train_report.summary()
        logging.info('\nValidation set:')
        val_report.summary()
        logging.info('\nTesting set:')
        test_report.summary()

        train_report.save(os.path.join(self.save_dir, 'train_set'))
        val_report.save(os.path.join(self.save_dir, 'validation_set'))
        test_report.save(os.path.join(self.save_dir, config['model_name']))

        self.checkpoint.summary()

        return None

    def _tune_model(self, model, config, dataset):
        iterations = config['iterations']
        batch_size = config['batch_size']

        loss_func = loss_select(config['loss_function'])
        loss_func = loss_func.to(self.device)

        self.loss_func = loss_func
        self.batch_size = batch_size

        optim = optim_select(model, config['optimizer'])

        iterations_per_epoch = math.ceil(len(dataset) / batch_size)
        epochs = math.ceil(iterations / iterations_per_epoch)

        iter_index, train_finish = 0, False
        for epoch in range(epochs):
            # training part
            dataset.set_mode('train')
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
            model, optim, iter_index, train_finish, _ = self._loop(
                    iter_index,
                    iterations,
                    model,
                    optim,
                    loss_func,
                    dataloader,
                    'train')

            # validation and testing
            dataset.set_mode('val')
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, batch_size = batch_size)
                with torch.no_grad():
                    model, optim, _iter_index, _train_finish, _ = self._loop(
                            iter_index,
                            iterations,
                            model,
                            optim,
                            loss_func,
                            dataloader,
                            'val')

            dataset.set_mode('test')
            dataloader = DataLoader(dataset, batch_size = batch_size)
            with torch.no_grad():
                model, optim, _iter_index, _train_finish, _ = self._loop(
                        iter_index,
                        iterations,
                        model,
                        optim,
                        loss_func,
                        dataloader,
                        'test')

        self.writer.close()

        return model

    def _loop(self,
            iter_index,
            iterations,
            model,
            optim,
            loss_func,
            dataloader,
            mode,
            summary = False):

        # the training, val and testing loop
        if mode == 'train':
            model = model.train().to(self.device)
        else:
            model = model.eval().to(self.device)

        train_finish = False
        total_labels = []
        total_outputs = []
        for iter, data in enumerate(dataloader):
            spectra, labels = data
            spectra = spectra.float().to(self.device)
            labels = labels.float().to(self.device)

            # init optimizer
            if mode == 'train':
                optim.zero_grad()

            outputs = model(spectra)

            # calculate loss of the model
            loss = loss_func(outputs, labels)
            # calculate gradient and update model
            if mode == 'train':
                loss.backward()

                if self.gradient_clipping is not None:
                    if self.gradient_clipping['method'].lower() == 'value':
                        torch.nn.utils.clip_grad_value_(model.parameters(),
                            clip_value = self.gradient_clipping['value'])
                    elif self.gradient_clipping['method'].lower() == 'norm':
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                            max_norm = self.gradient_clipping['value'])

                optim.step(iteration = iter_index)

            # summarize the information
            loss = loss.detach().cpu()
            total_labels.append(labels.cpu())
            total_outputs.append(outputs.detach().cpu())

            # record different stage
            if mode == 'train':
                iter_index += 1

                if iter_index % self.info_iter == 0:
                    logging.info('Progress: ' + str(iter_index) + ' / ' + str(iterations))

                self.writer.add_scalar('train_loss', loss, iter_index + 1)

                if iter_index >= iterations: 
                    train_finish = True
                    break

        total_outputs = torch.cat(total_outputs, dim = 0)
        total_outputs = (torch.sigmoid(total_outputs) > 0.5).float().numpy()
        total_labels = torch.cat(total_labels, dim = 0).numpy()

        report = ClassificationReport(total_outputs, total_labels,
                label_note = self.dataset.descriptions(), info = False)

        if mode == 'val':
            self.checkpoint.add_state(model, report, iter_index)

        if summary:
            report.summary()

        self.writer.add_scalar(mode + '_accuracy', report.accuracy(), iter_index)
        self.writer.add_scalar(mode + '_recall', report.recall(), iter_index)
        self.writer.add_scalar(mode + '_precision', report.precision(), iter_index)
        self.writer.add_scalar(mode + '_f1', report.f1(), iter_index)
        if mode == 'val' or mode == 'test':
            self.writer.add_scalar(mode + '_loss', report.loss(), iter_index)

        return model, optim, iter_index, train_finish, report


