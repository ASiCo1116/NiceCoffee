import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NIRDeepLearningModule

from ...utils import save_object, load_pickle_obj

#--------------------------------------------------------------------------------
#ann.py contain all feed forward neural network
#--------------------------------------------------------------------------------


__all__ = ['NIRANN']


class NIRANN(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            node = 900,
            layers = 3,
            drop = 0.4):

        super(NIRANN, self).__init__()

        if not isinstance(input_size, tuple):
            raise TypeError('Argumet: input_size only tuple is available.')

        if isinstance(output_size, int):
            self.output_size = output_size
            self.output_descriptions = None
        elif isinstance(output_size, (list, tuple)):
            self.output_size = len(output_size)
            self.output_descriptions = output_size
        else:
            raise TypeError('Only description list or int can be fed into model.')

        if not isinstance(model_name, str):
            raise TypeError('Argument: model_name only str available.')

        if not isinstance(node, int):
            raise TypeError('Argument: node must be a int.')

        if node < 1:
            raise ValueError('Argument: node must larger than zero.')

        if not isinstance(layers, int):
            raise TypeError('Argument: layer must be a int.')

        if layers < 1:
            raise ValueError('Argument: layer must larger than one.')

        if not isinstance(drop, float):
            raise TypeError('Argument: drop must be a float between 0 and 1.')

        self.input_size = input_size
        self.model_name = model_name
        self.node = node
        self.drop = drop
        self.layers = layers

        self.linears = nn.Sequential()
        self.linears.add_module('first_linear', nn.Linear(900, self.node))
        self.linears.add_module('first_dropout', nn.Dropout(p = self.drop))
        self.linears.add_module('first_relu', nn.ReLU())
        for i in range(self.layers - 2):
            self.linears.add_module('linear' + str(i + 1),  nn.Linear(self.node, self.node))
            self.linears.add_module('dropout' + str(i + 1), nn.Dropout(p = self.drop))
            self.linears.add_module('relu' + str(i + 1),  nn.ReLU())

        self.linears.add_module('last_linear', nn.Linear(self.node, self.output_size))

    def forward(self, input_data):
        x = torch.squeeze(input_data)
        x = self.linears(x)

        return x

    def __repr__(self):
        text = super(NIRANN, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRANN'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['node'] = self.node
        state['drop'] = self.drop
        state['layers'] = self.laeyrs
        state['preprocess_func'] = self.preprocess_func

        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRANN saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRANN only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRANN':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                node = state['node'],
                layers = state['layers'],
                drop = state['drop'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


