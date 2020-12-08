import os
import logging

from .classifier import AdaBoost, RF, SVM, KNN, PLSDA, DeepClassifier
from .nir_model import NIRDeepLearningModule, NIRANN, NIRVGG7, NIRVGG10, NIRVGG16, NIRResNet18, NIRResNet34, NIRResNet50, NIRResNet101, NIRResNet152, NIRSEResNet18, NIRSEResNet34, NIRSEResNet50, NIRSEResNet101, NIRSEResNet152

from ..utils import load_pickle_obj, init_torch_device

#--------------------------------------------------------------------------------
# property.py contains the extened properties for model object
#--------------------------------------------------------------------------------


__all__ = ['ModelExtensionDict', 'model_from_config', 'load_model', 'is_nir_model', 'init_torch_device']


ModelExtensionDict = {'.svm': SVM, '.rf': RF, '.adab': AdaBoost, '.knn': KNN, '.plsda': PLSDA, '.dlcls': DeepClassifier}

def model_from_config(config, directory = True):
    if isinstance(config, str):
        # load trained model from path
        if directory:
            extension_list = list(ModelExtensionDict.keys())
            file_list = os.listdir(config)
            found = False
            for f in file_list:
                extension = '.' + f.split('.')[-1]
                if extension in extension_list:
                    fname = os.path.join(config, f)
                    model =  ModelExtensionDict[extension]
                    found = True

                    if 'best_model' in f:
                        break

            if not found:
                raise OSError('Model can not be found in this directory.')
        else:
            fname = config
            model = ModelExtensionDict[('.' + config.split('.')[-1])]

        return model(input_size = (), output_size = 0).load(fname)

    elif isinstance(config, dict):
        if config['model']['model_type'].lower() == 'svm':
            model = ModelExtensionDict['.svm']
        elif config['model']['model_type'].lower() == 'rf':
            model = ModelExtensionDict['.rf']
        elif config['model']['model_type'].lower() == 'adaboost':
            model = ModelExtensionDict['.adab']
        elif config['model']['model_type'].lower() == 'knn':
            model = ModelExtensionDict['.knn']
        elif config['model']['model_type'].lower() == 'plsda':
            model = ModelExtensionDict['.plsda']
        elif config['model']['model_type'].lower() == 'dl_classifier':
            model = DeepClassifier
        elif config['model']['model_type'].lower() == 'nirann':
            model = NIRANN
        elif config['model']['model_type'].lower() == 'nirvgg7':
            model = NIRVGG7
        elif config['model']['model_type'].lower() == 'nirvgg10':
            model = NIRVGG10
        elif config['model']['model_type'].lower() == 'nirvgg16':
            model = NIRVGG16
        elif config['model']['model_type'].lower() == 'nirresnet18':
            model = NIRResNet18
        elif config['model']['model_type'].lower() == 'nirresnet34':
            model = NIRResNet34
        elif config['model']['model_type'].lower() == 'nirresnet50':
            model = NIRResNet50
        elif config['model']['model_type'].lower() == 'nirresnet101':
            model = NIRResNet101
        elif config['model']['model_type'].lower() == 'nirresnet152':
            model = NIRResNet152
        elif config['model']['model_type'].lower() == 'nirseresnet18':
            model = NIRSEResNet18
        elif config['model']['model_type'].lower() == 'nirseresnet34':
            model = NIRSEResNet34
        elif config['model']['model_type'].lower() == 'nirseresnet50':
            model = NIRSEResNet50
        elif config['model']['model_type'].lower() == 'nirseresnet101':
            model = NIRSEResNet101
        elif config['model']['model_type'].lower() == 'nirseresnet152':
            model = NIRSEResNet152
        else:
            raise ValueError(config['model']['model_type'], ' is not a valid model selection.')

        return model

    else:
        raise RuntimeError('The input file is not available to select model in this project.')

    return None

def load_model(path, device = None, init_device = True):
    state = load_pickle_obj(path, extension_check = False)
    try:
        preprocess_func = state['preprocess_func']
    except KeyError:
        from lib.utils import save_object
        state['preprocess_func'] = None
        file_paths = os.path.split(path)
        name, extension = file_paths[1].split('.')[0], file_paths[1].split('.')[1]
        save_object(os.path.join(file_paths[0], name), state, extension = extension)

    try:
        output_descriptions = state['output_descriptions']
    except KeyError:
        from lib.utils import save_object
        state['output_descriptions'] = state['output_size']
        file_paths = os.path.split(path)
        name, extension = file_paths[1].split('.')[0], file_paths[1].split('.')[1]
        save_object(os.path.join(file_paths[0], name), state, extension = extension)


    if state['model_type'] == 'SVM':
        model = SVM().load(path)
    elif state['model_type'] == 'RF':
        model = RF().load(path)
    elif state['model_type'] == 'AdaBoost':
        model = AdaBoost().load(path)
    elif state['model_type'] == 'KNN':
        model = KNN().load(path)
    elif state['model_type'] == 'PLSDA':
        model = PLSDA().load(path)
    else:
        if init_device:
            device = init_torch_device(device)

        if state['model_type'] == 'DeepClassifier':
            model = DeepClassifier()
        elif state['model_type'] == 'NIRANN':
            model = NIRANN()
        elif state['model_type'] == 'NIRVGG7':
            model = NIRVGG7()
        elif state['model_type'] == 'NIRVGG10':
            model = NIRVGG10()
        elif state['model_type'] == 'NIRVGG16':
            model = NIRVGG16()
        elif state['model_type'] == 'NIRResNet18':
            model = NIRResNet18()
        elif state['model_type'] == 'NIRResNet34':
            model = NIRResNet34()
        elif state['model_type'] == 'NIRResNet50':
            model = NIRResNet50()
        elif state['model_type'] == 'NIRResNet101':
            model = NIRResNet101()
        elif state['model_type'] == 'NIRResNet152':
            model = NIRResNet152()
        elif state['model_type'] == 'NIRSEResNet18':
            model = NIRSEResNet18()
        elif state['model_type'] == 'NIRSEResNet34':
            model = NIRSEResNet34()
        elif state['model_type'] == 'NIRSEResNet50':
            model = NIRSEResNet50()
        elif state['model_type'] == 'NIRSEResNet101':
            model = NIRSEResNet101()
        elif state['model_type'] == 'NIRSEResNet152':
            model = NIRSEResNet152()
        else:
            raise ValueError('Model is not implemented in the program, please check that the model was trained by this program.')

        model.load(path)
        if device is not None:
            model = model.to(device)

    return model, device

def is_nir_model(model):
    if isinstance(model, NIRDeepLearningModule):
        return True
    else:
        return False


