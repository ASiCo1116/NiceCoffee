import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression

import torch
import torch.nn as nn

from ..utils import save_object, load_pickle_obj

from .utils import EmptyFunction, PermuteLayer, RNNPermuteLayer, select_conv, select_norm, conv_argument_transform 
from .base import DeepLearningModule, ConcatenatedML, AllZeroModel
from .residual_block import BasicBlock, BottleBlock

#--------------------------------------------------------------------------------
# classifier.py contains the network which was used in classifier training, and it
# can classify mass spectrum of GCMS into multi-labeled description categories
#--------------------------------------------------------------------------------


__all__ = ['RF', 'SVM', 'DeepClassifier', 'ResNet']


class SVM(ConcatenatedML):
    def __init__(self,
            input_size = (),
            output_size = 0,
            kernel = 'rbf',
            model_name = '',
            batch = True,
            verbose = True):

        super(SVM, self).__init__(
                input_size = input_size,
                output_size = output_size,
                model_name = model_name,
                batch = batch,
                verbose = verbose)

        if not isinstance(kernel, str):
            raise TypeError('Argument: kernel must be str, please check sklearn API for SVC usage.')

        kernel_available_list = ['linear', 'poly', 'sigmoid', 'rbf']
        if kernel.lower() not in kernel_available_list:
            raise ValueError('Argument: kernel must in ' + str(kernel_available_list))

        self.kernel = kernel.lower()

        self.models = []
        for i in range(self.output_size):
            self.models.append(SVC(kernel = self.kernel, gamma = 'auto', verbose = False))

        self.optimized_properties = {
                'c': {
                'numeric': True,
                'available': None,
                'upper_bound': 1e25,
                'lower_bound': 1e2,
                'binary_digit_overflow': 6,
                'type_function': float},
                }

    def save(self, path):
        state = {}
        state['model_type'] = 'SVM'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['kernel'] = self.kernel
        state['model_name'] = self.model_name
        state['batch'] = self.batch
        state['preprocess_func'] = self.preprocess_func
        state['models'] = self.models

        save_object(path, state, mode = 'pickle', extension = 'svm')

        return None

    def load(self, fname):
        if not fname.endswith('.svm'):
            raise RuntimeError('lib.model.SVM only can load file endswith .svm which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'SVM':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.input_size = state['input_size']
        self.output_size = state['output_size']
        self.output_descriptions = state['output_descriptions']
        self.kernel = state['kernel']
        self.model_name = state['model_name']
        self.batch = state['batch']
        self.preprocess_func = state['preprocess_func']
        self.models = state['models']

        return self

    def _train(self, index, data, label, **kwargs):
        c = kwargs.get('c', 1.0)

        if np.sum(label) == 0:
            self.models[index] = AllZeroModel()
        else:
            self.models[index] = SVC(kernel = self.kernel, C = c, gamma = 'auto', verbose = False)
            self.models[index].fit(data, label)

        return None

    def __repr__(self):
        space_length = len(self.__class__.__name__) + 2

        properties_str = '( model_name: ' + str(self.model_name) + ',\n'
        properties_str +=  (' ' * space_length + 'input_size: ' + str(self.input_size) + ',\n')
        properties_str +=  (' ' * space_length + 'output_size: ' + str(self.output_size) + ',\n')
        properties_str +=  (' ' * space_length + 'kernel: ' + self.kernel + ',\n')
        properties_str +=  (' ' * space_length + 'batch: ' + str(self.batch) + ')')

        return self.__class__.__name__ + properties_str
    

class RF(ConcatenatedML):
    def __init__(self,
            criterion = 'entropy',
            input_size = (),
            output_size = 0,
            model_name = '',
            batch = True,
            verbose = True):

        super(RF, self).__init__(
                input_size = input_size,
                output_size = output_size,
                model_name = model_name,
                batch = batch,
                verbose = verbose)

        criterion_available_list = ['entropy', 'gini']
        if criterion not in criterion_available_list:
            raise ValueError('Argument: criterion must be entropy or gini.')

        self.criterion = criterion
        self.models = []
        for i in range(self.output_size):
            self.models.append(RandomForestClassifier(criterion = criterion))

        self.optimized_properties = {
                'tree_num': {
                'numeric': True,
                'available': None,
                'upper_bound': 1e3,
                'lower_bound': 11,
                'binary_digit_overflow': 3,
                'type_function': int},
                }

    def save(self, path):
        state = {}
        state['model_type'] = 'RF'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['criterion'] = self.criterion
        state['model_name'] = self.model_name
        state['batch'] = self.batch
        state['preprocess_func'] = self.preprocess_func
        state['models'] = self.models

        save_object(path, state, mode = 'pickle', extension = 'rf')

        return None

    def load(self, fname):
        if not fname.endswith('.rf'):
            raise RuntimeError('lib.model.RF only can load file endswith .rf which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'RF':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.input_size = state['input_size']
        self.output_size = state['output_size']
        self.output_descriptions = state['output_descriptions']
        self.criterion = state['criterion']
        self.model_name = state['model_name']
        self.batch = state['batch']
        self.preprocess_func = state['preprocess_func']
        self.models = state['models']


        return self

    def _train(self, index, data, label, **kwargs):
        tree_num = kwargs.get('tree_num', 10)
        if isinstance(tree_num, float):
             tree_num = int(tree_num)

        if np.sum(label) == 0:
            self.models[index] = AllZeroModel()
        else:
            self.models[index] = RandomForestClassifier(n_estimators = tree_num, criterion = self.criterion)
            self.models[index].fit(data, label)

        return None

    def __repr__(self):
        space_length = len(self.__class__.__name__) + 2

        properties_str = '( model_name: ' + str(self.model_name) + ',\n'
        properties_str +=  (' ' * space_length + 'input_size: ' + str(self.input_size) + ',\n')
        properties_str +=  (' ' * space_length + 'output_size: ' + str(self.output_size) + ',\n')
        properties_str +=  (' ' * space_length + 'criterion: ' + self.criterion + ',\n')
        properties_str +=  (' ' * space_length + 'batch: ' + str(self.batch) + ')')

        return self.__class__.__name__ + properties_str

        
class DeepClassifier(DeepLearningModule):

    # ----- structure candidates -----
    #  1. input -> ANN -> output
    #  2. input -> CNN feature extractor -> ANN -> output
    #  3. input -> CNN feature extractor -> RNN -> ANN -> output
    # --------------------------------
    # the last layer of the ANN is output size of the model
    #
    # ANN argument:
    #   layers_num: int
    #   hidden_size: list or tuple (output part of the layer, length must be layers_num - 1)
    #   dropout: float (only work when layers_num > 1)
    #
    # --------------------------------
    # CNN is the feature extractor of the model, and it is based on residual block
    # 
    # CNN argument:
    #   blocks_num: int
    #   block_selection: 'basic' or 'bottle'
    #   channels_num: list or tuple (length must be blocks_num + 1)
    #   conv_fsize: list or tuple (length must be the same as blocks_num)
    #   avg_pool: bool (only work if RNN is None)
    # --------------------------------
    # RNN can extract the feature through intention time
    #
    # RNN argument:
    #   select: str ('lstm', 'gru')
    #   layers_num: int
    #   hidden_size: int
    #   dropout: float
    #   bidirectional: bool
    #   avg_pool: bool (the data will be pool in the feature dim to reduce the affect of time dim.)
    #
    # --------------------------------

    def __init__(self,
            input_size = (),
            output_size = 0,
            model_name = '',
            cnn_dim = 'none',
            batch = True,
            verbose = False,
            ann = None,
            cnn = None,
            rnn = None):

        super(DeepClassifier, self).__init__(
                batch = batch,
                verbose = verbose)

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

        if cnn_dim not in ['1d', '2d', 'none']:
            raise ValueError("Argument: cnn_dim must be '1d' or '2d' or 'none'.")

        self.input_size = input_size
        self.model_name = model_name
        self.cnn_dim = cnn_dim

        if rnn is None:
            rnn_use = False
        else:
            rnn_use = True

        data_size = list(input_size)

        if cnn is None:
            self.cnn_argument = None
            self.cnn = cnn
        else:
            if cnn_dim == 'none':
                raise RuntimeError("Arguemnt: cnn_dim can't be 'none' when CNN is set in the DeepClassifier.")

            self.cnn_argument = cnn
            self.cnn, data_size = self._make_cnn(data_size, **cnn, rnn_use = rnn_use)

        if rnn is None:
            self.rnn_argument = None
            self.rnn = rnn
        else:
            self.rnn_argument = rnn
            self.rnn, data_size = self._make_rnn(data_size, **rnn)

        if ann is None:
            self.ann_argument = None
            self.ann = self._make_ann(data_size)
        else:
            self.ann_argument = ann
            self.ann = self._make_ann(data_size, **ann)

    def forward(self, x):
        # batch first forward
        if self.cnn is not None:
            if self.cnn_dim == '1d':
                x = x.permute(0, 2, 1)
            elif self.cnn_dim == '2d':
                x = x.unsqueeze(1)

            x = self.cnn(x)

            if self.cnn_dim == '1d':
                x = x.permute(0, 2, 1)
            elif self.cnn_dim == '2d':
                x = x.permute(0, 2, 1, 3)
                sample_num, retention_time = x.size(0), x.size(1)
                x = x.reshape(sample_num, retention_time, -1)

        if self.rnn is not None:
            x = self.rnn(x)

        sample_num = x.size(0)
        x = x.reshape(sample_num, -1)
        x = self.ann(x)

        return x

    def __repr__(self):
        text = super(DeepClassifier, self).__repr__()

        text += '\nOther properties:\n  model_name: ' + str(self.model_name) + ',\n'
        text += '  input_size: ' + str(self.input_size) + ',\n'
        text += '  output_size: ' + str(self.output_size) + ',\n'
        text += '  total_parameters: ' + str(self.count_params()) + '\n'

        return text

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'DeepClassifier'
        state['input_size'] = self.input_size
        state['output_size'] = self.output_size
        state['output_descriptions'] = self.output_descriptions
        state['model_name'] = self.model_name
        state['cnn_dim'] = self.cnn_dim
        state['cnn'] = self.cnn_argument
        state['rnn'] = self.rnn_argument
        state['ann'] = self.ann_argument
        state['preprocess_func'] = self.preprocess_func
        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'dlcls')

        return None

    def load(self, fname):
        if not fname.endswith('.dlcls'):
            raise RuntimeError('lib.model.DeepClassifier only can load file endswith .adab which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'DeepClassifier':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                input_size = state['input_size'],
                output_size = state['output_descriptions'],
                model_name = state['model_name'],
                cnn_dim = state['cnn_dim'],
                cnn = state['cnn'],
                rnn = state['rnn'],
                ann = state['ann'])

        self.preprocess_func = state['preprocess_func']

        self.load_state_dict(state['state_dict'])

        return self

    def _make_cnn(self,
            data_size,
            blocks_num = 0,
            block_selection = 'basic',
            channels_num = (),
            conv_fsize = (),
            avg_pool = False,
            rnn_use = False):

        if block_selection.lower() not in ['basic', 'bottle']:
            raise ValueError("Argument: block_selection must be 'basic' or 'bottle'.")

        if len(channels_num) != (blocks_num + 1):
            raise RuntimeError('Argument: channels_num must be the list or tuple which length is blocks_num + 1.')

        if len(conv_fsize) != (blocks_num + 1):
            raise RuntimeError('Argument: conv_fsize must be the list or tuple which length is blocks_num + 1.')

        conv_fsize[0] = conv_argument_transform(conv_fsize[0], self.cnn_dim)
        if self.cnn_dim == '1d':
            data_size[1] = channels_num[1]
            data_size[0] = self._cnn_data_size(data_size[0], conv_fsize[0])
            in_channel = self.input_size[1]
            lead_conv = nn.Conv1d
            lead_bn = nn.BatchNorm1d
            fsize = conv_fsize[0]
            padding = conv_fsize[0] // 2
        elif self.cnn_dim == '2d':
            data_size.insert(0, channels_num[0])
            data_size[1] = self._cnn_data_size(data_size[1], conv_fsize[0][0])
            data_size[2] = self._cnn_data_size(data_size[2], conv_fsize[0][1])
            in_channel = 1
            lead_conv = nn.Conv2d
            lead_bn = nn.BatchNorm2d
            fsize = conv_fsize[0]
            padding = (conv_fsize[0][0] // 2, conv_fsize[0][1] // 2)

        cnn = [lead_conv(in_channel, channels_num[0], kernel_size = fsize, padding = padding, bias = False),
                lead_bn(channels_num[0]),
                nn.ReLU(inplace = True)]

        for i in range(blocks_num):
            conv_fsize[i + 1] = conv_argument_transform(conv_fsize[i + 1], self.cnn_dim)
            if self.cnn_dim == '1d':
                data_size[1] = channels_num[i + 1]
                data_size[0] = self._cnn_data_size(data_size[0], conv_fsize[i + 1])
            elif self.cnn_dim == '2d':
                data_size[0] = channels_num[i + 1]
                data_size[1] = self._cnn_data_size(data_size[1], conv_fsize[i + 1][0])
                data_size[2] = self._cnn_data_size(data_size[2], conv_fsize[i + 1][1])

            if channels_num[i] != channels_num[i + 1]:
                fsize = conv_argument_transform(1, self.cnn_dim)
                stride = conv_argument_transform(1, self.cnn_dim)
                short = nn.Sequential(
                        select_conv(self.cnn_dim, channels_num[i], channels_num[i + 1], fsize, stride),
                        select_norm(self.cnn_dim, channels_num[i + 1]),
                        )
            else:
                short = None

            if block_selection == 'basic':
                cnn.append(BasicBlock(channels_num[i], channels_num[i + 1],
                        dim = self.cnn_dim,
                        conv_fsize = conv_fsize[i + 1],
                        short = short))

            elif block_selection == 'bottle':
                cnn.append(BottleBlock(channels_num[i], channels_num[i + 1],
                        dim = self.cnn_dim,
                        conv_fsize = conv_fsize[i + 1],
                        short = short))

        if not rnn_use and avg_pool:
            if self.cnn_dim == '1d':
                cnn.append(nn.AdaptiveAvgPool1d(1))
                data_size[0] = 1
            elif self.cnn_dim == '2d':
                cnn.append(nn.AdaptiveAvgPool2d((1, 1)))
                data_size[1], data_size[2] = 1, 1
        elif rnn_use and avg_pool:
            raise RuntimeError('Argument: avg_pool in CNN can not set to True when RNN is in the DeepClassifier.')
                
        if self.cnn_dim == '2d':
            data_size = [data_size[1], data_size[0] * data_size[2]]

        return nn.Sequential(*cnn), data_size

    def _cnn_data_size(self, old, kernel_size, dilation = 1, stride = 1):
        padding = kernel_size // 2
        new = ((old + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        return int(new)

    def _make_rnn(self,
            data_size,
            select = 'lstm',
            layers_num = 0,
            hidden_size = 0,
            dropout = 0.,
            bidirectional = False,
            avg_pool = True):

        rnn = []
        if select == 'lstm':
            rnn.append(nn.LSTM(input_size = data_size[1],
                    hidden_size = hidden_size,
                    num_layers = layers_num,
                    batch_first = True,
                    dropout = dropout,
                    bidirectional = bidirectional))
        elif select == 'gru':
            rnn.append(nn.GRU(input_size = data_size[1],
                    hidden_size = hidden_size,
                    num_layers = layers_num,
                    batch_first = True,
                    dropout = dropout,
                    bidirectional = bidirectional))

        direction = 2 if bidirectional else 1
        data_size[1] = int(hidden_size * direction)

        if avg_pool:
            rnn.append(RNNPermuteLayer((0, 2, 1)))
            rnn.append(nn.AdaptiveAvgPool1d(1))
            rnn.append(PermuteLayer((0, 2, 1)))

            data_size[0] = 1

        return nn.Sequential(*rnn), data_size

    def _make_ann(self,
            data_size,
            layers_num = 1,
            hidden_size = (),
            dropout = 0.):

        if len(hidden_size) != (layers_num - 1):
            raise RuntimeError('Argument: hidden_size in ANN must be the same as layers_num - 1')

        data_size = np.prod(data_size).astype(np.int)
        if layers_num == 1:
            ann = nn.Linear(data_size, self.output_size)
        else:
            ann = []
            for i in range(len(hidden_size)):
                ann.append(nn.Linear(data_size, hidden_size[i]))
                ann.append(nn.Dropout(p = dropout))
                ann.append(nn.ReLU(inplace = True))
                data_size = hidden_size[i]

            ann.append(nn.Linear(data_size, self.output_size))
            ann = nn.Sequential(*ann)

        return ann


class ResNet(DeepLearningModule):
    def __init__(self,
            blocks,
            layers,
            conv_fsize,
            layer_dim = '1d',
            channel = 64,
            input_size = (),
            output_size = 0,
            model_name = '',
            cnn_dim = None,
            norm_layer = None,
            avg_pool = True,
            batch = True,
            verbose = False):

        super(ResNet, self).__init__(
                batch = batch,
                verbose = verbose)

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

        if layer_dim not in ['1d', '2d', 'none']:
            raise ValueError("Argument: layer_dim must be '1d' or '2d' or 'none'.")

        self.input_size = input_size
        self.model_name = model_name
        self.layer_dim = layer_dim

    def save(self):
        pass

    def load(self):
        pass

    def forward(self, x):
        pass
