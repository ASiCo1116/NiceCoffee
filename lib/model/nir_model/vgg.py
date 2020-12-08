import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NIRDeepLearningModule

from ...utils import save_object, load_pickle_obj

#--------------------------------------------------------------------------------
#vgg.py contain all vgg base model
#--------------------------------------------------------------------------------


__all__ = ['NIRVGG7', 'NIRVGG10', 'NIRVGG16']


class NIRVGG7(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            conv_fsize = 3,
            channel = 128,
            node = 1024,
            dropout = 0.2):

        super(NIRVGG7, self).__init__()
        #to inherit the parameter of the model

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

        if not isinstance(channel, int):
            raise TypeError('Argument: channel must be a int.')

        if channel < 1:
            raise ValueError('Argument: channel must at least be 1.')

        if not isinstance(node, int):
            raise TypeError('Argument: node must be a int.')

        if node < 1:
            raise ValueError('Argument: node must larger than zero.')

        if not isinstance(dropout, float):
            raise TypeError('Argument: dropout must be a float between 0 and 1.')

        if dropout > 1 or dropout < 0:
            raise ValueError('Argument: dropout must be a float between 0 and 1.')

        self.input_size = input_size
        self.model_name = model_name
        self.dropout = dropout
        self.node = node
        self.conv_fsize = conv_fsize
        self.channel = channel

        self.conv1_1 = nn.Conv1d(1, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_1 = nn.BatchNorm1d(self.channel)
        self.conv1_2 = nn.Conv1d(self.channel, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_2 = nn.BatchNorm1d(self.channel)
        self.max1 = nn.MaxPool1d(kernel_size = 2)

        self.conv2_1 = nn.Conv1d(self.channel, (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_1 = nn.BatchNorm1d(self.channel * 2)
        self.conv2_2 = nn.Conv1d((self.channel * 2), (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_2 = nn.BatchNorm1d(self.channel * 2)
        self.max2 = nn.MaxPool1d(kernel_size = 2)

        self.dense1 = nn.Linear((225 * (self.channel * 2)), self.node)
        self.dropout1 = nn.Dropout(p = self.dropout)
        self.dense2 = nn.Linear(self.node, self.node)
        self.dropout2 = nn.Dropout(p = self.dropout)
        self.dense3 = nn.Linear(self.node, self.output_size)

    def forward(self, input_data):
        x = self.bn1_1(F.relu(self.conv1_1(input_data)))
        x = self.bn1_2(F.relu(self.conv1_2(x)))
        x = self.max1(x)

        x = self.bn2_1(F.relu(self.conv2_1(x)))
        x = self.bn2_2(F.relu(self.conv2_2(x)))
        x = self.max2(x)

        x = x.view(-1, self._flatten(x))
        x = self.dropout1(F.relu(self.dense1(x)))
        x = self.dropout2(F.relu(self.dense2(x)))

        return x

    def _flatten(self, x):
        size = x.size()[1:]#all feature dimension of the data
        feature_num = 1
        for s in size:
            feature_num *= s

        return feature_num

    def __repr__(self):
        text = super(NIRVGG7, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRVGG7'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.channel
        state['conv_fsize'] = self.conv_fsize
        state['node'] = self.node
        state['dropout'] = self.dropout
        state['preprocess_func'] = self.preprocess_func

        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRVGG7 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRVGG7 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRVGG7':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize = state['conv_fsize'],
                node = state['node'],
                dropout = state['dropout'])

        state['preprocess_func'] = self.preprocess_func

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRVGG10(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            conv_fsize = 3,
            channel = 128,
            node = 1024,
            dropout = 0.2):

        super(NIRVGG10, self).__init__()
        #to inherit the parameter of the model

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

        if not isinstance(channel, int):
            raise TypeError('Argument: channel must be a int.')

        if channel < 1:
            raise ValueError('Argument: channel must at least be 1.')

        if not isinstance(node, int):
            raise TypeError('Argument: node must be a int.')

        if node < 1:
            raise ValueError('Argument: node must larger than zero.')

        if not isinstance(dropout, float):
            raise TypeError('Argument: dropout must be a float between 0 and 1.')

        if dropout > 1 or dropout < 0:
            raise ValueError('Argument: dropout must be a float between 0 and 1.')

        self.input_size = input_size
        self.model_name = model_name
        self.dropout = dropout
        self.node = node
        self.conv_fsize = conv_fsize
        self.channel = channel

        self.conv1_1 = nn.Conv1d(1, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_1 = nn.BatchNorm1d(self.channel)
        self.conv1_2 = nn.Conv1d(self.channel, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_2 = nn.BatchNorm1d(self.channel)
        self.max1 = nn.MaxPool1d(kernel_size = 2)

        self.conv2_1 = nn.Conv1d(self.channel, (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_1 = nn.BatchNorm1d(self.channel * 2)
        self.conv2_2 = nn.Conv1d((self.channel * 2), (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_2 = nn.BatchNorm1d(self.channel * 2)
        self.max2 = nn.MaxPool1d(kernel_size = 2)

        self.conv3_1 = nn.Conv1d((self.channel * 2), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_1 = nn.BatchNorm1d(self.channel * 4)
        self.conv3_2 = nn.Conv1d((self.channel * 4), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_2 = nn.BatchNorm1d(self.channel * 4)
        self.conv3_3 = nn.Conv1d((self.channel * 4), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_3 = nn.BatchNorm1d(self.channel * 4)
        self.max3 = nn.MaxPool1d(kernel_size = 5)

        self.dense1 = nn.Linear((45 * (self.channel * 4)), self.kernel)
        self.dropout1 = nn.Dropout(p = self.dropout)
        self.dense2 = nn.Linear(self.kernel, self.kernel)
        self.dropout2 = nn.Dropout(p = self.dropout)
        self.dense3 = nn.Linear(self.kernel, self.output_size)

    def forward(self, input_data):
        x = self.bn1_1(F.relu(self.conv1_1(input_data)))
        x = self.bn1_2(F.relu(self.conv1_2(x)))
        x = self.max1(x)

        x = self.bn2_1(F.relu(self.conv2_1(x)))
        x = self.bn2_2(F.relu(self.conv2_2(x)))
        x = self.max2(x)

        x = self.bn3_1(F.relu(self.conv3_1(x)))
        x = self.bn3_2(F.relu(self.conv3_2(x)))
        x = self.bn3_3(F.relu(self.conv3_3(x)))
        x = self.max3(x)

        x = x.view(-1, self._flatten(x))
        x = self.dropout1(F.relu(self.dense1(x)))
        x = self.dropout2(F.relu(self.dense2(x)))

        return x

    def _flatten(self, x):
        size = x.size()[1:]#all feature dimension of the data
        feature_num = 1
        for s in size:
            feature_num *= s

        return feature_num

    def __repr__(self):
        text = super(NIRVGG10, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRVGG10'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['model_name'] = self.model_name
        state['channel'] = self.channel
        state['conv_fsize'] = self.conv_fsize
        state['node'] = self.node
        state['dropout'] = self.dropout
        state['preprocess_func'] = self.preprocess_func

        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRVGG10 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRVGG10 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRVGG10':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_size'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize = state['conv_fsize'],
                node = state['node'],
                dropout = state['dropout'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRVGG16(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            conv_fsize = 3,
            channel = 128,
            node = 1024,
            dropout = 0.2):

        super(NIRVGG16, self).__init__()
        #to inherit the parameter of the model

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

        if not isinstance(channel, int):
            raise TypeError('Argument: channel must be a int.')

        if channel < 1:
            raise ValueError('Argument: channel must at least be 1.')

        if not isinstance(node, int):
            raise TypeError('Argument: node must be a int.')

        if node < 1:
            raise ValueError('Argument: node must larger than zero.')

        if not isinstance(dropout, float):
            raise TypeError('Argument: dropout must be a float between 0 and 1.')

        if dropout > 1 or dropout < 0:
            raise ValueError('Argument: dropout must be a float between 0 and 1.')

        self.input_size = input_size
        self.model_name = model_name
        self.dropout = dropout
        self.node = node
        self.conv_fsize = conv_fsize
        self.channel = channel

        self.conv1_1 = nn.Conv1d(1, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_1 = nn.BatchNorm1d(self.channel)
        self.conv1_2 = nn.Conv1d(self.channel, self.channel, kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn1_2 = nn.BatchNorm1d(self.channel)
        self.max1 = nn.MaxPool1d(kernel_size = 2)

        self.conv2_1 = nn.Conv1d(self.channel, (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_1 = nn.BatchNorm1d(self.channel * 2)
        self.conv2_2 = nn.Conv1d((self.channel * 2), (self.channel * 2), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn2_2 = nn.BatchNorm1d(self.channel * 2)
        self.max2 = nn.MaxPool1d(kernel_size = 2)

        self.conv3_1 = nn.Conv1d((self.channel * 2), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_1 = nn.BatchNorm1d(self.channel * 4)
        self.conv3_2 = nn.Conv1d((self.channel * 4), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_2 = nn.BatchNorm1d(self.channel * 4)
        self.conv3_3 = nn.Conv1d((self.channel * 4), (self.channel * 4), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn3_3 = nn.BatchNorm1d(self.channel * 4)
        self.max3 = nn.MaxPool1d(kernel_size = 3)

        self.conv4_1 = nn.Conv1d((self.channel * 4), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn4_1 = nn.BatchNorm1d(self.channel * 8)
        self.conv4_2 = nn.Conv1d((self.channel * 8), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn4_2 = nn.BatchNorm1d(self.channel * 8)
        self.conv4_3 = nn.Conv1d((self.channel * 8), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn4_3 = nn.BatchNorm1d(self.channel * 8)
        self.max4 = nn.MaxPool1d(kernel_size = 3)

        self.conv5_1 = nn.Conv1d((self.channel * 8), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn5_1 = nn.BatchNorm1d(self.channel * 8)
        self.conv5_2 = nn.Conv1d((self.channel * 8), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn5_2 = nn.BatchNorm1d(self.channel * 8)
        self.conv5_3 = nn.Conv1d((self.channel * 8), (self.channel * 8), kernel_size = self.conv_fsize,
                padding = int(self.conv_fsize / 2), bias = False)
        self.bn5_3 = nn.BatchNorm1d(self.channel * 8)
        self.max5 = nn.MaxPool1d(kernel_size = 5)

        self.dense1 = nn.Linear((5 * (self.channel * 8)), self.kernel)
        self.dropout1 = nn.Dropout(p = self.dropout)
        self.dense2 = nn.Linear(self.kernel, self.kernel)
        self.dropout2 = nn.Dropout(p = self.dropout)
        self.dense3 = nn.Linear(self.kernel, self.output_size)

    def forward(self, input_data):
        x = self.bn1_1(F.relu(self.conv1_1(input_data)))
        x = self.bn1_2(F.relu(self.conv1_2(x)))
        x = self.max1(x)

        x = self.bn2_1(F.relu(self.conv2_1(x)))
        x = self.bn2_2(F.relu(self.conv2_2(x)))
        x = self.max2(x)

        x = self.bn3_1(F.relu(self.conv3_1(x)))
        x = self.bn3_2(F.relu(self.conv3_2(x)))
        x = self.bn3_3(F.relu(self.conv3_3(x)))
        x = self.max3(x)

        x = self.bn4_1(F.relu(self.conv4_1(x)))
        x = self.bn4_2(F.relu(self.conv4_2(x)))
        x = self.bn4_3(F.relu(self.conv4_3(x)))
        x = self.max4(x)

        x = self.bn5_1(F.relu(self.conv5_1(x)))
        x = self.bn5_2(F.relu(self.conv5_2(x)))
        x = self.bn5_3(F.relu(self.conv5_3(x)))
        x = self.max5(x)

        x = x.view(-1, self._flatten(x))
        x = self.dropout1(F.relu(self.dense1(x)))
        x = self.dropout2(F.relu(self.dense2(x)))
        x = self.dense3(x)

        return x

    def _flatten(self, x):
        size = x.size()[1:]#all feature dimension of the data
        feature_num = 1
        for s in size:
            feature_num *= s

        return feature_num

    def __repr__(self):
        text = super(NIRVGG16, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRVGG16'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['model_name'] = self.model_name
        state['channel'] = self.channel
        state['conv_fsize'] = self.conv_fsize
        state['node'] = self.node
        state['dropout'] = self.dropout
        state['preprocess_func'] = self.preprocess_func

        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRVGG16 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRVGG16 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRVGG16':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_size'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize = state['conv_fsize'],
                node = state['node'],
                dropout = state['dropout'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


