import os
import json
import yaml
import pickle
import logging

import torch
import torch.cuda as cuda

#--------------------------------------------------------------------------------
# utils contains some function used in mutiple process in the project
#--------------------------------------------------------------------------------


__all__ = ['load_config', 'init_torch_device', 'save_object', 'load_pickle_obj',
        'load_json_obj', 'transform_empty_argument', 'init_directory']

def load_config(fname):
    f = open(fname)
    content = yaml.load(f, Loader = yaml.FullLoader)
    f.close()
    return content

def init_directory(path):
    assert isinstance(path, str)

    if not os.path.isdir(path):
        os.makedirs(path)

    return path

def init_torch_device(select = None, multi_gpu = False):
    # select calculating device
    if isinstance(select, torch.device):
        return select

    if multi_gpu:
        if not isinstance(select, list):
            raise TypeError('Device selection must be a int list or tuple to derive multi-GPU programming.')

        select.sort(key = lambda x: x)
        if select < 0:
            raise ValueError('Device selection for multi-GPU must be index of CUDA device, negative index means CPU.')

        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:' + str(select[0]))
        device_note = ''
        for index in select:
            device_note += 'cuda:' + str(index) + ', '

        device_note = device_note[: -2]
        logging.info('Hardware setting done, using device: ' + device_note)
    else:
        if select is None:
           if cuda.is_available():
               torch.backends.cudnn.benchmark = True
               cuda.set_device(0)
               device = torch.device('cuda:' + str(0))
               logging.info('Hardware setting done, using device: cuda:' + str(0))
           else:
               device = torch.device('cpu')
               logging.info('Hardware setting done, using device: cpu')
        else:
            logging.info('Init calculating device ...')
            if select < 0:
                device = torch.device('cpu')
                logging.info('Hardware setting done, using device: cpu')
            else:
                torch.backends.cudnn.benchmark = True
                cuda.set_device(select)
                device = torch.device('cuda:' + str(select))
                logging.info('Hardware setting done, using device: cuda:' + str(select))

    return device

def save_object(fname, obj, mode = 'pickle', extension = None):
    # the function is used to save some data in class or object in .pkl file
    # or json file
    if not isinstance(fname, str):
        raise TypeError(fname, "is not string object, it can't be the saving name of file.")

    if mode == 'pickle':
        if extension is None:
            if not fname.endswith('.pkl'):
                fname += '.pkl'

        else:
            if not isinstance(extension, str):
                raise TypeError('File extension must be a str object.')

            fname = fname + '.' + extension

        with open(fname, 'wb') as out_file:
            pickle.dump(obj, out_file)

    elif mode == 'json':
        if not fname.endswith('.json'):
            fname += '.json'

        obj = json.dumps(obj)
        with open(fname, 'w') as out_file:
            out_file.write(obj) 

    out_file.close()

    return None

def load_pickle_obj(fname, extension_check = True):
    # the function is used to read the data in saved in pickle format
    if extension_check:
        if not fname.endswith('.pkl'):
            raise RuntimeError(fname, 'is not a pickle file.')

    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

    return None

def load_json_obj(fname):
    # the functionis used to read the data in .json file
    if not fname.endswith('.json'):
        raise RuntimeError(fname, 'is not a json file.')

    with open(fname, 'r') as in_file:
        return json.loads(in_file.read())

    return None

def transform_empty_argument(config, argument_name):
    try:
        argument = config[argument_name]
        if argument is None:
            argument = {}

    except KeyError:
        argument = {}

    return argument


