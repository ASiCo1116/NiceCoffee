import logging

import numpy as np

import torch
import torch.nn as nn

from ..utils import save_object, load_pickle_obj

from .utils import numpy_instance_check
from .base import Report

#--------------------------------------------------------------------------------
# classification_report.py contains the evaluation metric for odor classification
# all metrics were designed for multi-label classification
#--------------------------------------------------------------------------------

class ClassificationReport(Report):
    def __init__(self,
            prediction,
            label,
            label_note = None,
            block_length = 15,
            terminal_length = 120,
            eps = 1e-7,
            percentage = True,
            info = True):
        # label and prediction must be a 2d-numpy array
        # label note must has same length of class numbers in label and prediciton
        self.prediction = numpy_instance_check(prediction, dim = 2)
        self.label = numpy_instance_check(label, dim = 2)

        if self.prediction.shape != self.label.shape:
            raise RuntimeError('The prediction array and target array must has same shape.')

        if label_note is not None:
            if not isinstance(label_note, (list, tuple)):
                raise TypeError('Argument: label_note must be a list or tuple.')

            if len(label_note) != label.shape[1]:
                raise RuntimeError('The length of label_note should has same shape of class numbers')

        if not isinstance(block_length, int):
            raise TypeError('Argument: block_length must be a int.')

        if not isinstance(terminal_length, int):
            raise TypeError('Argument: terminal_length must be a int.')

        if not isinstance(eps, float):
            raise TypeError('Argument: eps must be a float.')

        if eps >= 1e-4:
            logging.warning('Too big eps will affect the results of ClassificationReport.')

        if not isinstance(percentage, bool):
            raise TypeError('Argument: precentage must be a boolean.')

        if not isinstance(info, bool):
            raise TypeError('Argument: info must be a boolean.')

        self.label_note = label_note
        self.label_num = self.label.shape[1]
        self.block_length = block_length
        self.terminal_length = terminal_length
        self.block_num = self.terminal_length // self.block_length
        self.eps = eps
        self.percentage = percentage
        self.info = info
        self.report_type = 'ClassificationReport'

        self.report = self._eval(self.prediction, self.label)

        if info:
            logging.info('Successfully generate classification report.')

    def save(self, path):
        state = {}
        state['type'] = 'ClassificationReport'
        state['prediction'] = self.prediction
        state['label'] = self.label
        state['label_note'] = self.label_note
        state['label_num'] = self.label_num
        state['precentage'] = self.percentage
        state['report'] = self.report

        save_object(path, state, mode = 'pickle', extension = 'report')

        if self.info:
            logging.info('Classification report saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.report'):
            raise RuntimeError('lib.eval.ClassificationReport only can load file endswith .report.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['type'] != 'ClassificationReport':
            raise RuntimeError('The object is not a state file which is saved by ClassificationReport.')

        self.prediction = state['prediction']
        self.label = state['label']
        self.label_note = state['label_note']
        self.label_num = state['label_num']
        self.percentage = state['precentage']
        self.report = state['report']

        if self.info:
            logging.info('Load state file success !!')

        return self

    def __repr__(self):
        space_length = len(self.__class__.__name__) + 2

        properties_str = '( label_note: ' + str(self.label_note) + ',\n'
        properties_str +=  (' ' * space_length + 'precentage: ' + str(self.percentage) + ')') 

        return self.__class__.__name__ +  properties_str

    def summary(self):
        logging.info('=' * self.terminal_length)
        logging.info('Summary:')

        logging.info('Recall for each class:')
        self._summary_class_property('recall')

        logging.info('Percision for each class:')
        self._summary_class_property('precision')

        logging.info('Accuracy for each class:')
        self._summary_class_property('accuracy')

        logging.info('TP for each class:')
        self._summary_class_property('tp', ratio = False)

        logging.info('FP for each class:')
        self._summary_class_property('fp', ratio = False)

        logging.info('FN for each class:')
        self._summary_class_property('fn', ratio = False)

        logging.info('TN for each class:')
        self._summary_class_property('tn', ratio = False)

        if self.percentage:
            properties_summary = 'Accuracy: %.2f | Recall: %.2f | Percision: %.2f | F1 score: %.3f | Loss % .6f' %\
                    (self.report['accuracy']['total'] * 100, self.report['recall']['total'] * 100,
                    self.report['precision']['total'] * 100, self.report['f1']['total'], self.report['loss']['total'])
        else:
            properties_summary = 'Accuracy: %.2f | Recall: %.2f | Percision: %.2f | F1 score: %.3f | Loss % .6f' %\
                    (self.report['accuracy']['total'], self.report['recall']['total'],
                    self.report['precision']['total'], self.report['f1']['total'], self.report['loss']['tatal'])

        logging.info(properties_summary)

        logging.info('=' * self.terminal_length)

        return None

    def get_index(self, criterion):
        if criterion not in list(self.report.keys()):
            raise RuntimeError(criterion, ' is not a valid index for ClassificationReport.')

        return self.report[criterion]['total']

    def f1(self):
        return self.report['f1']['total']

    def loss(self):
        return self.report['loss']['total']

    def recall(self):
        return self.report['recall']['total']

    def recalls(self):
        return self.report['recall']['class']

    def precision(self):
        return self.report['precision']['total']

    def precisions(self):
        return self.report['precision']['class']

    def accuracy(self):
        return self.report['accuracy']['total']

    def accuracies(self):
        return self.report['accuracy']['class']

    def accuracys(self):
        return self.accuracies()

    def tps(self):
        return self.report['tp']['class']

    def fps(self):
        return self.report['fp']['class']

    def tns(self):
        return self.report['tn']['class']

    def fns(self):
        return self.report['fn']['class']

    def _eval(self, prediction, label):
        loss_func = nn.BCELoss()
        loss = loss_func(torch.tensor(prediction).float(), torch.tensor(label).float())
        loss = loss.detach().numpy()

        class_recall_total = np.sum(label, axis = 0) # TP + FN
        class_recall_error = np.sum(np.maximum((label - prediction), 0), axis = 0) # FN
        class_recall = (1 - class_recall_error / (class_recall_total + self.eps))
        recall_total = np.sum(class_recall_total) 
        recall_error = np.sum(class_recall_error)
        recall = (1 - recall_error / (recall_total + self.eps))

        class_fn = class_recall_error.copy()
        class_tp = class_recall_total - class_fn

        class_precision_total = np.sum(prediction, axis = 0) # TP + FP
        class_precision_error =  np.sum(np.maximum((prediction - label), 0), axis = 0) # FP
        class_precision = (1 - class_precision_error / (class_precision_total + self.eps))
        precision_total = np.sum(class_precision_total)
        precision_error = np.sum(class_precision_error)
        precision = (1 - precision_error / (precision_total + self.eps))

        class_fp = class_precision_error.copy()
        class_tn = label.shape[0] - (class_fn + class_tp + class_fp)

        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        class_accuracy_total = np.sum(np.ones_like(label), axis = 0)
        class_accuracy_error = np.sum(np.abs(label - prediction), axis = 0)
        class_accuracy = (1 - class_accuracy_error / (class_accuracy_total + self.eps))
        accuracy_total = np.sum(class_accuracy_total)
        accuracy_error = np.sum(class_accuracy_error)
        accuracy = (1 - accuracy_error / (accuracy_total + self.eps))

        report = {
                'loss': {'total': loss},
                'f1': {'total': f1},
                'recall': {'class': class_recall, 'total': recall},
                'precision': {'class': class_precision, 'total': precision},
                'accuracy': {'class': class_accuracy, 'total': accuracy},
                'tp': {'class': class_tp},
                'fp': {'class': class_fp},
                'tn': {'class': class_tn},
                'fn': {'class': class_fn},
                }

        return report

    def _summary_class_property(self, select, ratio = True):
        lines = ['', '']
        for i in range(self.label_num):
            if self.label_note is None:
                lines[0] += f'Class {i + 1}'.rjust(self.block_length)
            else:
                lines[0] += self.label_note[i].rjust(self.block_length)

            if ratio:
                if self.percentage:
                    lines[1] += ('%.2f' % float(self.report[select]['class'][i] * 100) + '%').rjust(self.block_length)
                else:
                    lines[1] += ('%.2f' % float(self.report[select]['class'][i])).rjust(self.block_length)

            else:
                lines[1] += ('{0}'.format(self.report[select]['class'][i])).rjust(self.block_length)

            if i != 0 and i % (self.block_num - 1) == 0:
                logging.info(lines[0])
                logging.info(lines[1])
                lines = ['', '']
            if i == self.label_num - 1:
                logging.info(lines[0])
                logging.info(lines[1])

        return None


