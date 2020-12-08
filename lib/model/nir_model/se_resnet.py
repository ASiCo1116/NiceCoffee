import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NIRDeepLearningModule

from ...utils import save_object, load_pickle_obj

#--------------------------------------------------------------------------------
#se_resnet.py contain the model of squeeze and extraction block.
#the based of the mdoel is residual network.
#--------------------------------------------------------------------------------


__all__ = ['NIRSEModule', 'NIRSEBasicBlock', 'NIRSEBottleBlock', 'NIRSEResNet18',
        'NIRSEResNet34', 'NIRSEResNet50', 'NIRSEResNet101', 'NIRSEResNet152']


class NIRSEModule(nn.Module):
    def __init__(self,
            channels,
            reduction):

        super(NIRSEModule, self).__init__()

        self.channels = channels
        self.reduction = reduction

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace = True)

        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size = 1, padding = 0)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size = 1, padding = 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origin = x

        x = self.avg_pool(x)
     
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return torch.mul(origin, x)


class NIRSEBasicBlock(nn.Module):
    def __init__(self,
            in_channel,
            out_channel,
            reduction,
            conv_fsize,
            stride = 1,
            short = None):

        super(NIRSEBasicBlock, self).__init__()

        self.conv_fsize = conv_fsize
        self.stride = stride
        self.short = short

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size = self.conv_fsize,
                padding = self.conv_fsize // 2)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size = self.conv_fsize,
                padding = self.conv_fsize // 2)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.se_block = NIRSEModule(channels = out_channel, reduction = reduction)

    def forward(self, input_data):
        residual = input_data

        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.short is not None:
            residual = self.short(input_data)

        x = self.se_block(x) + residual

        return x


class NIRSEBottleBlock(nn.Module):
    def __init__(self,
            in_channel, 
            out_channel,
            expansion,
            reduction,
            conv_fsize_one,
            conv_fsize_two,
            stride = 1,
            short = None):

        super(NIRSEBottleBlock, self).__init__()

        self.expansion = expansion
        self.conv_fsize_one = conv_fsize_one
        self.conv_fsize_two = conv_fsize_two
        self.stride = stride
        self.short = short

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size = self.conv_fsize_one,
                padding = self.conv_fsize_one // 2)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size = self.conv_fsize_two,
                padding = self.conv_fsize_two // 2)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = nn.Conv1d(out_channel, out_channel * self.expansion, kernel_size = self.conv_fsize_one,
                padding = self.conv_fsize_one // 2)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace = True)

        self.se_block = NIRSEModule(channels = out_channel * self.expansion, reduction = reduction)

    def forward(self, input_data):
        residual = input_data

        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.short is not None:
            residual = self.short(input_data)

        x = self.se_block(x) + residual

        return x


class NIRSEResNet18(NIRDeepLearningModule):
    #this part is the main part of the SEResNet model
    #this SEResNet model have 18 layer
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            channel = 64,
            conv_fsize = 3,
            reduction = 2,
            avg_pool = True):

        super(NIRSEResNet18, self).__init__()

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
            raise ValueError('Argument: channel must at least larger than one.')

        if not isinstance(conv_fsize, int):
            raise TypeError('Argument: conv_fsize must be a int.')

        if conv_fsize < 1:
            raise ValueError('Argument: conv_fsize must at least larger than one.')

        if not isinstance(reduction, int):
            raise TypeError('Argument: reduction must be a int.')

        if reduction < 1:
            raise ValueError('Argument: reduction must at least be one.')

        if not isinstance(avg_pool, bool):
            raise TypeError('Argument: avg_pool must be a boolean.')

        self.input_size = input_size
        self.model_name = model_name
        self.channel = channel
        self.conv_fsize = conv_fsize
        self.reduction = reduction
        self.avg_pool = avg_pool
        self.relu = nn.ReLU(inplace = True)

        self.lead_conv = nn.Conv1d(1, self.channel, kernel_size = int(self.conv_fsize * 2 + 1),
                padding = self.conv_fsize, bias = False)
        self.lead_bn = nn.BatchNorm1d(self.channel)
        self.max = nn.MaxPool1d(kernel_size = 2)

        self.layer1 = self._make_layer(self.channel, self.channel, block_num = 2,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer2 = self._make_layer(self.channel, self.channel * 2, block_num = 2,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer3 = self._make_layer(self.channel * 2, self.channel * 4, block_num = 2,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer4 = self._make_layer(self.channel * 4, self.channel * 8, block_num = 2,
                conv_fsize = self.conv_fsize, reduction = self.reduction)

        if avg_pool:
            self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(self.channel * 8, self.output_size)
        else:
            self.dense = nn.Linear(450 * (self.channel * 8), self.output_size)

    def _make_layer(self, in_channel, out_channel, block_num, conv_fsize, reduction, stride = 1):
        short = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm1d(out_channel)
                )

        layers = []
        layers.append(NIRSEBasicBlock(in_channel, out_channel, conv_fsize = conv_fsize, reduction = reduction, 
                stride = stride, short = short))
        for i in range(1, block_num):
            layers.append(NIRSEBasicBlock(out_channel, out_channel, conv_fsize = conv_fsize, reduction = reduction))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x = self.lead_conv(input_data)
        x = self.lead_bn(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = self.avg_pool_layer(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def __repr__(self):
        text = super(NIRSEResNet18, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRSEResNet18'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.channel
        state['conv_fsize'] = self.conv_fsize
        state['reduction'] = self.reduction
        state['avg_pool'] = self.avg_pool
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRSEResNet18 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRSEResNet18 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRSEResNet18':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize = state['conv_fsize'],
                reduction = state['reduction'],
                avg_pool = state['avg_pool'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRSEResNet34(NIRDeepLearningModule):
    #this part is the main part of the SEResNet model
    #this SEResNet model have 34 layer
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            channel = 64,
            conv_fsize = 3,
            reduction = 2,
            avg_pool = True):

        super(NIRSEResNet34, self).__init__()

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
            raise ValueError('Argument: channel must at least larger than one.')

        if not isinstance(conv_fsize, int):
            raise TypeError('Argument: conv_fsize must be a int.')

        if conv_fsize < 1:
            raise ValueError('Argument: conv_fsize must at least larger than one.')

        if not isinstance(reduction, int):
            raise TypeError('Argument: reduction must be a int.')

        if reduction < 1:
            raise ValueError('Argument: reduction must at least be one.')

        if not isinstance(avg_pool, bool):
            raise TypeError('Argument: avg_pool must be a boolean.')

        self.input_size = input_size
        self.model_name = model_name
        self.channel = channel
        self.conv_fsize = conv_fsize
        self.reduction = reduction
        self.avg_pool = avg_pool
        self.relu = nn.ReLU(inplace = True)

        self.lead_conv = nn.Conv1d(1, self.channel, kernel_size = int(self.conv_fsize * 2 + 1),
                padding = self.conv_fsize, bias = False)
        self.lead_bn = nn.BatchNorm1d(self.channel)
        self.max = nn.MaxPool1d(kernel_size = 2)

        self.layer1 = self._make_layer(self.channel, self.channel, block_num = 3,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer2 = self._make_layer(self.channel, self.channel * 2, block_num = 4,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer3 = self._make_layer(self.channel * 2, self.channel * 4, block_num = 6,
                conv_fsize = self.conv_fsize, reduction = self.reduction)
        self.layer4 = self._make_layer(self.channel * 4, self.channel * 8, block_num = 3,
                conv_fsize = self.conv_fsize, reduction = self.reduction)

        if avg_pool:
            self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(self.channel * 8, self.output_size)
        else:
            self.dense = nn.Linear(450 * (self.channel * 8), self.output_size)

    def _make_layer(self, in_channel, out_channel, block_num, conv_fsize, reduction, stride = 1):
        short = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm1d(out_channel)
                )

        layers = []
        layers.append(NIRSEBasicBlock(in_channel, out_channel, conv_fsize = conv_fsize, reduction = reduction,
                stride = stride, short = short))
        for i in range(1, block_num):
            layers.append(NIRSEBasicBlock(out_channel, out_channel, conv_fsize = conv_fsize, reduction = reduction))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x = self.lead_conv(input_data)
        x = self.lead_bn(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = self.avg_pool_layer(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def __repr__(self):
        text = super(NIRSEResNet34, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRSEResNet34'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.channel
        state['conv_fsize'] = self.conv_fsize
        state['reduction'] = self.reduction
        state['avg_pool'] = self.avg_pool
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRSEResNet34 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRSEResNet34 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRSEResNet34':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize = state['conv_fsize'],
                reduction = state['reduction'],
                avg_pool = state['avg_pool'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRSEResNet50(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            channel = 64,
            conv_fsize_one = 1,
            conv_fsize_two = 3,
            reduction = 2,
            expansion = 4,
            avg_pool = True):

        super(NIRSEResNet50, self).__init__()

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
            raise ValueError('Argument: channel must at least larger than one.')

        if not isinstance(conv_fsize_one, int):
            raise TypeError('Argument: conv_fsize_one must be a int.')

        if conv_fsize_one < 1:
            raise ValueError('Argument: conv_fsize_one must at least larger than one.')

        if not isinstance(conv_fsize_two, int):
            raise TypeError('Argument: conv_fsize_two must be a int.')

        if conv_fsize_two < 1:
            raise ValueError('Argument: conv_fsize_two must at least larger than one.')

        if not isinstance(reduction, int):
            raise TypeError('Argument: reduction must be a int.')

        if reduction < 1:
            raise ValueError('Argument: reduction must at least be one.')

        if not isinstance(expansion, int):
            raise TypeError('Argument: expansion must be a int.')

        if expansion < 1:
            raise ValueError('Argument: expansion must larger than one.')

        if not isinstance(avg_pool, bool):
            raise TypeError('Argument: avg_pool must be a boolean.')

        self.input_size = input_size
        self.model_name = model_name
        self.channel = channel
        self.out_channel = channel
        self.conv_fsize_one = conv_fsize_one
        self.conv_fsize_two = conv_fsize_two
        self.reduction = reduction
        self.expansion = expansion
        self.avg_pool = avg_pool

        self.lead_conv = nn.Conv1d(1, self.channel, kernel_size = int(self.conv_fsize_two * 2 + 1),
                padding = self.conv_fsize_two, bias = False)
        self.lead_bn = nn.BatchNorm1d(self.channel)
        self.max = nn.MaxPool1d(kernel_size = 2)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(self.out_channel, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer2 = self._make_layer(self.out_channel * 2, block_num = 4, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer3 = self._make_layer(self.out_channel * 4, block_num = 6, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer4 = self._make_layer(self.out_channel * 8, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)

        if avg_pool:
            self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(self.channel, self.output_size)
        else:
            self.dense = nn.Linear(450 * (self.out_channel * self.expansion * 8), self.output_size)

    def _make_layer(self, out_channel, block_num, expansion, conv_fsize_one, conv_fsize_two, reduction, stride = 1):
        short = None
        if stride != 1 or self.channel != out_channel * expansion:
            short = nn.Sequential(
                    nn.Conv1d(self.channel, out_channel * expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm1d(out_channel * expansion)
                    )

        layers = []
        layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion, conv_fsize_one = conv_fsize_one,
            conv_fsize_two = conv_fsize_two, reduction = reduction, stride = stride, short = short))
        self.channel = out_channel * expansion
        for i in range(1, block_num):
            layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x = self.lead_conv(input_data)
        x = self.lead_bn(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = self.avg_pool_layer(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def __repr__(self):
        text = super(NIRSEResNet50, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRSEResNet50'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.out_channel
        state['conv_fsize_one'] = self.conv_fsize_one
        state['conv_fsize_two'] = self.conv_fsize_two
        state['reduction'] = self.reduction
        state['expansion'] = self.expansion
        state['avg_pool'] = self.avg_pool
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRSEResNet50 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRSEResNet50 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRSEResNet50':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize_one = state['conv_fsize_one'],
                conv_fsize_two = state['conv_fsize_two'],
                reduction = state['reduction'],
                expansion = state['expansion'],
                avg_pool = state['avg_pool'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRSEResNet101(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            channel = 64,
            conv_fsize_one = 1,
            conv_fsize_two = 3,
            reduction = 2,
            expansion = 4,
            avg_pool = True):

        super(NIRSEResNet101, self).__init__()

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
            raise ValueError('Argument: channel must at least larger than one.')

        if not isinstance(conv_fsize_one, int):
            raise TypeError('Argument: conv_fsize_one must be a int.')

        if conv_fsize_one < 1:
            raise ValueError('Argument: conv_fsize_one must at least larger than one.')

        if not isinstance(conv_fsize_two, int):
            raise TypeError('Argument: conv_fsize_two must be a int.')

        if conv_fsize_two < 1:
            raise ValueError('Argument: conv_fsize_two must at least larger than one.')

        if not isinstance(reduction, int):
            raise TypeError('Argument: reduction must be a int.')

        if reduction < 1:
            raise ValueError('Argument: reduction must at least be one.')

        if not isinstance(expansion, int):
            raise TypeError('Argument: expansion must be a int.')

        if expansion < 1:
            raise ValueError('Argument: expansion must larger than one.')

        if not isinstance(avg_pool, bool):
            raise TypeError('Argument: avg_pool must be a boolean.')

        self.input_size = input_size
        self.model_name = model_name
        self.channel = channel
        self.out_channel = channel
        self.conv_fsize_one = conv_fsize_one
        self.conv_fsize_two = conv_fsize_two
        self.reduction = reduction
        self.expansion = expansion
        self.avg_pool = avg_pool

        self.lead_conv = nn.Conv1d(1, self.channel, kernel_size = int(self.conv_fsize_two * 2 + 1),
                padding = self.conv_fsize_two, bias = False)
        self.lead_bn = nn.BatchNorm1d(self.channel)
        self.max = nn.MaxPool1d(kernel_size = 2)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(self.out_channel, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer2 = self._make_layer(self.out_channel * 2, block_num = 4, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer3 = self._make_layer(self.out_channel * 4, block_num = 23, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer4 = self._make_layer(self.out_channel * 8, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)

        if avg_pool:
            self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(self.channel, self.output_size)
        else:
            self.dense = nn.Linear(450 * (self.out_channel * self.expansion * 8), self.output_size)

    def _make_layer(self, out_channel, block_num, expansion, conv_fsize_one, conv_fsize_two, reduction, stride = 1):
        short = None
        if stride != 1 or self.channel != out_channel * expansion:
            short = nn.Sequential(
                    nn.Conv1d(self.channel, out_channel * expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm1d(out_channel * expansion)
                    )

        layers = []
        layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion, conv_fsize_one = conv_fsize_one,
            conv_fsize_two = conv_fsize_two, reduction = reduction, stride = stride, short = short))
        self.channel = out_channel * expansion
        for i in range(1, block_num):
            layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x = self.lead_conv(input_data)
        x = self.lead_bn(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = self.avg_pool_layer(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def __repr__(self):
        text = super(NIRSEResNet101, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRSEResNet101'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.out_channel
        state['conv_fsize_one'] = self.conv_fsize_one
        state['conv_fsize_two'] = self.conv_fsize_two
        state['reduction'] = self.reduction
        state['expansion'] = self.expansion
        state['avg_pool'] = self.avg_pool
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRSEResNet101 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRSEResNet101 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRSEResNet101':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize_one = state['conv_fsize_one'],
                conv_fsize_two = state['conv_fsize_two'],
                reduction = state['reduction'],
                expansion = state['expansion'],
                avg_pool = state['avg_pool'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


class NIRSEResNet152(NIRDeepLearningModule):
    def __init__(self,
            input_size = (),
            output_size = 21,
            model_name = '',
            channel = 64,
            conv_fsize_one = 1,
            conv_fsize_two = 3,
            reduction = 2,
            expansion = 4,
            avg_pool = True):

        super(NIRSEResNet152, self).__init__()

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
            raise ValueError('Argument: channel must at least larger than one.')

        if not isinstance(conv_fsize_one, int):
            raise TypeError('Argument: conv_fsize_one must be a int.')

        if conv_fsize_one < 1:
            raise ValueError('Argument: conv_fsize_one must at least larger than one.')

        if not isinstance(conv_fsize_two, int):
            raise TypeError('Argument: conv_fsize_two must be a int.')

        if conv_fsize_two < 1:
            raise ValueError('Argument: conv_fsize_two must at least larger than one.')

        if not isinstance(reduction, int):
            raise TypeError('Argument: reduction must be a int.')

        if reduction < 1:
            raise ValueError('Argument: reduction must at least be one.')

        if not isinstance(expansion, int):
            raise TypeError('Argument: expansion must be a int.')

        if expansion < 1:
            raise ValueError('Argument: expansion must larger than one.')

        if not isinstance(avg_pool, bool):
            raise TypeError('Argument: avg_pool must be a boolean.')

        self.input_size = input_size
        self.model_name = model_name
        self.channel = channel
        self.out_channel = channel
        self.conv_fsize_one = conv_fsize_one
        self.conv_fsize_two = conv_fsize_two
        self.reduction = reduction
        self.expansion = expansion
        self.avg_pool = avg_pool

        self.lead_conv = nn.Conv1d(1, self.channel, kernel_size = int(self.conv_fsize_two * 2 + 1),
                padding = self.conv_fsize_two, bias = False)
        self.lead_bn = nn.BatchNorm1d(self.channel)
        self.max = nn.MaxPool1d(kernel_size = 2)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(self.out_channel, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer2 = self._make_layer(self.out_channel * 2, block_num = 8, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer3 = self._make_layer(self.out_channel * 4, block_num = 36, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)
        self.layer4 = self._make_layer(self.out_channel * 8, block_num = 3, expansion = self.expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction)

        if avg_pool:
            self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(self.channel, self.output_size)
        else:
            self.dense = nn.Linear(450 * (self.out_channel * self.expansion * 8), self.output_size)

    def _make_layer(self, out_channel, block_num, expansion, conv_fsize_one, conv_fsize_two, reduction, stride = 1):
        short = None
        if stride != 1 or self.channel != out_channel * expansion:
            short = nn.Sequential(
                    nn.Conv1d(self.channel, out_channel * expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm1d(out_channel * expansion)
                    )

        layers = []
        layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion, conv_fsize_one = conv_fsize_one,
            conv_fsize_two = conv_fsize_two, reduction = reduction, stride = stride, short = short))
        self.channel = out_channel * expansion
        for i in range(1, block_num):
            layers.append(NIRSEBottleBlock(self.channel, out_channel, expansion = expansion,
                conv_fsize_one = conv_fsize_one, conv_fsize_two = conv_fsize_two, reduction = reduction))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x = self.lead_conv(input_data)
        x = self.lead_bn(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = self.avg_pool_layer(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def __repr__(self):
        text = super(NIRSEResNet152, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'NIRSEResNet152'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['channel'] = self.out_channel
        state['conv_fsize_one'] = self.conv_fsize_one
        state['conv_fsize_two'] = self.conv_fsize_two
        state['reduction'] = self.reduction
        state['expansion'] = self.expansion
        state['avg_pool'] = self.avg_pool
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')
        logging.info('Model: NIRSEResNet152 saving done.')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.NIRSEResNet152 only can load file endswith .dlcls which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'NIRSEResNet152':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                channel = state['channel'],
                conv_fsize_one = state['conv_fsize_one'],
                conv_fsize_two = state['conv_fsize_two'],
                reduction = state['reduction'],
                expansion = state['expansion'],
                avg_pool = state['avg_pool'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self


