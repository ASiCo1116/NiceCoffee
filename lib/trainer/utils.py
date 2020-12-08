import os
import yaml
import logging

import torch
import torch.cuda as cuda

from torch.optim import Adam, Adagrad, Adamax, ASGD, RMSprop, SGD

from ..utils import init_torch_device, load_config, init_directory
from ..model import BaseModule
from ..eval import Report

#--------------------------------------------------------------------------------
# utils.py contains the function and object used by trainer and the ML/DL trainer
# selecting function.
#--------------------------------------------------------------------------------


__all__ = ['CriterionProperty', 'init_torch_device', 'load_config', 'init_directory',
        'optim_select', 'Checkpoint', 'LRScheduler']

CriterionProperty = {
        'accuracy': 'high',
        'recall': 'high',
        'precision': 'high',
        'f1': 'high',
        'loss': 'low',
        }

def optim_select(model, optim_config):
    try:
        argument_dict = optim_config['argument']
    except KeyError:
        argument_dict = {}

    try:
        schedule = optim_config['lr_schedule']
    except KeyError:
        schedule = {}

    if optim_config['select'].lower() == 'adam':
        optimizer = Adam(model.parameters(), **argument_dict)
    elif optim_config['select'].lower() == 'adagrad':
        optimizer = Adagrad(model.parameters(), **argument_dict)
    elif optim_config['select'].lower() == 'adamax':
        optimizer = Adamax(model.parameters(), **argument_dict)
    elif optim_config['select'].lower() == 'asgd':
        optimizer = ASGD(model.parameters(), **argument_dict)
    elif optim_config['select'].lower() == 'rmsprop':
        optimizer = RMSprop(model.parameters(), **argument_dict)
    elif optim_config['select'].lower() == 'sgd':
        optimizer = SGD(model.parameters(), **argument_dict)
    else:
        raise ValueError(optim_config['select'], ' is not valid in the training program.')

    scheduled_optim = LRScheduler(
            optimizer = optimizer,
            schedule = schedule)

    return scheduled_optim


class Checkpoint(object):
    def __init__(self,
        save_dir,
        criterion,
        start = 0,
        end = float('inf'),
        mode = 'delete'):
        self.save_dir = init_directory(save_dir)

        if criterion not in list(CriterionProperty.keys()):
            raise ValueError(criterion, ' is not valid for checkpoint.')

        self.mode_list = ['delete', 'append']
        if mode not in self.mode_list:
            raise ValueError(mode, ' is not valid for checkpoint.')

        if not isinstance(start, (int, float)):
            raise TypeError('Argument: start must be a int or float.')

        if not isinstance(end, (int, float)):
            raise TypeError('Argument: end must be a int or float.')

        if end < start:
            raise ValueError('Argument: end must bigger than argument: start.')

        self.start = start
        self.end = end
        self.criterion = criterion
        self.mode = mode
        self.preprocess_func = None
        self.best_score = None
        self.best_state = None
        self.best_model = None
        self.best_report = None

    def summary(self):
        # please use this method in the final part of training
        # the best model will be saved

        logging.info('State: ' + str(self.best_state) + ' is the best model !!')
        logging.info('The best ' + self.criterion + ' of the model: ' + str(self.best_score))

        if self.best_score is None or self.best_state is None or self.best_model is None or self.best_report is None:
            logging.info('No model insert into checkpoint, skip writing file process.')
            logging.info('Note that the ckeckpoint only insert in val loop,')
            logging.info('  if val_ratio is zero, checkpoint will not store any model and report.')
        else:
            self._write()
            logging.info('Best model and report was saved sucessfully !!')

        return None

    def add_state(self, model, report, state):
        if not isinstance(model, BaseModule):
            raise TypeError('Model must be the object inherited aroma_net.lib.model.BaseModule.')

        if not isinstance(report, Report):
            raise TypeError('Model must be the object inherited aroma_net.lib.eval.Report.')

        if not isinstance(state, (str, int, float)):
            raise TypeError('Argument: state must be a int or str.')

        new_score = report.get_index(self.criterion)
        if self.best_score is None or self.best_state is None or self.best_model is None or self.best_report is None:
            self.best_score = new_score
            self.best_state = state
            self.best_model = model
            self.best_report = report

            self._write()

            return None

        else:
            if type(state) != str:
                if state < self.start or state > self.end:
                    return None

            if CriterionProperty[self.criterion] == 'high':
                if new_score >= self.best_score:
                    self.best_score = new_score
                    self.best_state = state
                    self.best_model = model
                    self.best_report = report

                    self._write()

                    return None

            elif CriterionProperty[self.criterion] == 'low':
                if new_score <= self.best_score:
                    self.best_score = new_score
                    self.best_state = state
                    self.best_model = model
                    self.best_report = report

                    self._write()

                    return None

        return None

    def _write(self):
        if self.mode == 'delete':
            self.best_model.set_preprocess(self.preprocess_func)
            self.best_model.save(os.path.join(self.save_dir, 'best_state'))
            self.best_report.save(os.path.join(self.save_dir, 'best_report'))
        elif self.mode == 'append':
            self.best_model.set_preprocess(self.preprocess_func)
            self.best_model.save(os.path.join(self.save_dir, 'state_' + str(self.best_state)))
            self.best_report.save(os.path.join(self.save_dir, 'report_' + str(self.best_state)))

        return None


class LRScheduler(object):
    def __init__(self,
            optimizer,
            schedule,
            iteration = 0):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('Argument: optimizer must be a well-set torch.optim optimizer.')

        if not isinstance(schedule, dict):
            raise TypeError('Argument: scheduler must be a dict.')

        new_schedule = {}
        for obj in list(schedule.keys()):
            new_schedule[int(obj)] = schedule[obj]

        try:
            init_lr = new_schedule[iteration]
        except KeyError:
            for param_group in optimizer.param_groups:
                init_lr = param_group['lr']

            new_schedule[iteration] = init_lr

        if not isinstance(iteration, int):
            raise TypeError('Argument: iteration must be a positive int.')

        if iteration < 0:
            raise ValueError('Argument: iteration must be a positive int.')

        self.optimizer = optimizer
        self.schedule = new_schedule
        self.iteration = iteration

        self._adjust_iterations = list(self.schedule.keys())
        self._adjust_iterations.sort()
        self.lr_now = self._search_adjust_iterations(self.iteration, self._adjust_iterations)

    def zero_grad(self):
        self.optimizer.zero_grad()

        return None

    def step(self, iteration = None):
        if iteration is None:
            self.iteration = iteration

        self._adjust_lr(self.iteration)
        self.optimizer.step()
        self.iteration += 1

        return None

    def catch_lr(self):
        return self.lr_now

    def catch_iteration(self):
        return self.iteration

    def set_iteration(self, iteration):
        if not isinstance(iteration, int):
            raise TypeError('Argument: iteration must be a positive int.')

        if iteration < 0:
            raise ValueError('Argument: iteration must be a positive int.')

        self.iteration = iteration

        return None

    def _adjust_lr(self, iteration):
        if iteration in self._adjust_iterations:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.schedule[iteration]

        return None

    def _search_adjust_iterations(self, iteration, _adjust_iterations):
        ancher = _adjust_iterations[0]
        for iter in _adjust_iterations:
            if iteration >= iter:
                ancher = iter
                break

        lr = _adjust_iterations[ancher]

        return lr


