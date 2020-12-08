import logging

import numpy as np

import sklearn
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

from .base import TransformFunctionObject

#--------------------------------------------------------------------------------
# the gcms_preprocess.py is used in the odor model of the coffee odor recognition
# the gcms_preprocess.py contain all the function used in project of coffee GCMS
# the object in this file is to serialize the preprocess methods into GCMS model
#--------------------------------------------------------------------------------


__all__ = ['GCMSPreprocess', 'GCMSDatatype', 'GCMSReduceBlankSample', 'GCMSFlatten',
        'GCMSReduceDimension']


class GCMSPreprocess(TransformFunctionObject):
    def __init__(self, preprocess, batch = True):
        if not isinstance(preprocess, (list, tuple)):
            raise TypeError('Argument: preprocess must be a list or tuple.')

        if not isinstance(batch, bool):
            raise TypeError('Argument: batch must be a bool.')

        self.preprocess = self._init_preprocess(preprocess)
        self.batch = batch

    def __call__(self, spectra):
        for p in self.preprocess:
            spectra = p(spectra)

        return spectra

    def _init_preprocess(self, preprocess_list):
        for p in preprocess_list:
            if not isinstance(p, TransformFunctionObject):
                raise TypeError('GCMSPreperocess only can feed in object in aroma_net.data.transform.gcms_preprocess.')

        return preprocess_list

    def add_blank(self, blank_spectrum, situation):
        success = False
        for p in self.preprocess_func:
            if isinstance(p, GCMSReduceBlankSample):
                p.add_blank(blank_spectrum, situation)
                success = True

        if not success:
            logging.warning('GCMSPreprocess do not contain GCMSReduceBlankSample method, add_blank fail !!!')

        return None

    def add_pca(self, pca):
        success = False
        for p in self.preprocess_func:
            if isinstance(p, GCMSReduceDimension):
                p.add_pca(pca)
                success = True

        if not success:
            logging.warning('GCMSPreprocess do not contain GCMSReduceDiemsion method, add_pca fail !!!')

        return None

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)

        format_string += '\n)'
        format_string += '\nThis GCMSPreprocess object was used for old API, transform function is recommended.'
        return format_string


class GCMSDatatype(TransformFunctionObject):
    def __init__(self, datatype, batch = True):
        if not isinstance(datatype, str):
            raise TypeError('Argument: datatype must be ia string.')

        if datatype.lower() not in ['1d', '1d_rts', '1d_mzs', '2d']:
            raise ValueError(str(datatype) + 'is invalid setting for MassSpectrumDataset.datatype, [1d (equals to 1d_MZS), 1d_RTS (sum at retention time axis), 1d_MZS (sum at M/Z axis) or 2d].')

        if not isinstance(batch, bool):
            raise TypeError('Argument: batch must be a bool.')

        self.datatype = datatype.lower()
        self.batch = batch

    def __call__(self, spectra, return_type = 'origin'):
        if return_type.lower() not in ['origin', 'numpy', 'torch']:
            raise ValueError('Argument, dtype must be str:origin, str:numpy or str:torch.')

        if isinstance(spectra, np.ndarray):
            dim = len(spectra.shape)
            if self.batch:
                if dim != 3:
                    raise RuntimeError('Input data must be 3d array (batch, retention_time, m/z).')

                if self.datatype == '1d':
                    spectra = np.sum(spectra, axis = 2)
                elif self.datatype == '1d_mzs':
                    spectra = np.sum(spectra, axis = 2)
                elif self.datatype == '1d_rts':
                    spectra = np.sum(spectra, axis = 1)

            else:
                if dim != 2:
                    raise RuntimeError('Input data must be 2d array (retention_time, m/z).')

                if self.datatype == '1d':
                    spectra = np.sum(spectra, axis = 1)
                elif self.datatype == '1d_mzs':
                    spectra = np.sum(spectra, axis = 1)
                elif self.datatype == '1d_rts':
                    spectra = np.sum(spectra, axis = 0)

        elif isinstance(spectrum, torch.Tensor):
            dim = len(spectra.size())
            if self.batch:
                if dim != 3:
                    raise RuntimeError('Input data must be 3d tensor (batch, retention_time, m/z).')

                if self.datatype == '1d':
                    spectra = torch.sum(spectra, dim = 1)
                elif self.datatype == '1d_mzs':
                    spectra = torch.sum(spectra, dim = 1)
                elif self.datatype == '1d_rts':
                    spectra = torch.sum(spectra, dim = 1)

            elif dim == 2:
                if dim != 2:
                    raise RuntimeError('Input data must be 2d tensor (retention_time, m/z).')

                if self.datatype == '1d':
                    spectra = torch.sum(spectra, dim = 1)
                elif self.datatype == '1d_mzs':
                    spectra = torch.sum(spectra, dim = 1)
                elif self.datatype == '1d_rts':
                    spectra = torch.sum(spectra, dim = 0)

        else:
            raise TypeError('Input type error, method can not work.')

        if dim == 1:
            if self.datatype == '2d':
                raise RuntimeError('Input data can not be 1d if GCMSDatatype.datatype is set to 2d.')
            else:
                logging.warning('The data is already a 1d data, data will not be transformed in GCMSDatatype.') 

        if return_type.lower() == 'numpy':
            spectra = np.array(spectra)
        elif return_type.lower() == 'torch':
            spectra = torch.tensor(spectra)

        return spectra

    def __repr__(self):
        return self.__class__.__name__ + '(datatype={0})\n'.format(self.datatype)


class GCMSReduceBlankSample(TransformFunctionObject):
    def __init__(self,
            blank_samples = {}):

        if not isinstance(blank_samples, dict):
            raise TypeError('Argument: blank_samples must be a dict ')

        self.blank_samples = blank_samples
        if len(self.blank_samples) == 0:
            self.spectrum_size = None
        else:
            temp_size = None
            for key in self.blank_samples:
                temp_spectrum = self.blank_samples[key]
                if temp_size is None:
                    if isinstance(temp_spectrum, np.ndarray):
                        temp_size = temp_spectrum.shape
                    elif isinstance(temp_spectrum, torch.Tensor):
                        temp_size = temp_spectrum.size()
                    else:
                        raise TypeError('Blank sample in blank_samples must be a numpy.ndarray or torch.Tensor.')
                else:
                    if isinstance(temp_spectrum, np.ndarray):
                        if temp_size != temp_spectrum.shape:
                            raise RuntimeError('Numpy array size mismatch in blank sample dict.')
                    elif isinstance(temp_spectrum, torch.Tensor):
                        if temp_size != temp_spectrum.size():
                            raise RuntimeError('torch.Tensor size mismatch in blank sample dict.')

            self.spectrum_size = temp_size

    def __call__(self, spectra, blank_sample = None, situation = None):
        if blank_sample is None and situation is None:
            logging.warning('blank_sample and blank_sample situation not set, module will output original mass spectrum.')

        if blank_sample is not None:
            assert type(spectra) == type(blank_sample)
            spectra -= blank_sample
            return spectra

        elif situation is not None:
            if isinstance(spectra, np.ndarray):
                spectra -= np.array(self.blank_samples[situation])
            elif isinstance(spectrum, torch.Tensor):
                spectra -= torch.tensor(self.blank_samples[situation], device = spectra.device)

            return spectra

        else:
            raise RuntimeError('Argument: blank_sample and argument: situation can not be set together.')

    def add_blank(self, blank_spectrum, situation):
        if isinstance(blank_spectrum, np.ndarray):
            assert blank_spectrum.shape == self.spectrum_size
        elif isinstance(blank_spectrum, torch.Tensor):
            assert blank_spsectrum.size() == self.spectrum_size
        else:
            raise TypeError('Blank spectrum added must be a numpy.ndarray or torch.Tensor.')

        self.blank_samples[situation] = blank_spectrum

        return None

    def __repr__(self):
        return self.__class__.__name__ + '()\n'


class GCMSFlatten(TransformFunctionObject):
    def __init__(self, batch = True):
        if not isinstance(batch, bool):
            raise TypeError('Argument: batch msut be a bool.')

        self.batch = batch

    def __call__(self, spectra, return_type = 'origin'):
        if isinstance(spectra, np.ndarray):
            sample_num = spectra.shape[0]
        elif isinstance(spectrum, torch.Tensor):
            sample_num = spectra.size()[0]
        else:
            raise TypeError('Only numpy.ndarray or torch.Tensor can be fed into this GCMSFlatten object.')

        if return_type.lower() not in ['origin', 'numpy', 'torch']:
            raise ValueError('Argument: return_type must be str:origin, str:numpy or str:torch')

        if self.batch:
            spectra = spectra.reshape(sample_num, -1)
        else:
            spectra = spectra.reshape(-1)

        if return_type == 'numpy':
            spectra = np.array(spectra)
        elif return_type == 'torch':
            spectra = torch.tensor(spectra)

        return spectra

    def __repr__(self):
        return self.__class__.__name__ + '()\n'


class GCMSReduceDimension(TransformFunctionObject):
    def __init__(self,
            type = 'mass_spectrum',
            dim = None,
            dilated_number = None,
            mean_number = None,
            stride = None,
            retain_intensity = 'none',
            unit_vector = True,
            batch = True):
        
        if not isinstance(type, str):
            raise TypeError('Argument: type must be a str.')

        if type.lower() not in ['mass_spectrum', 'whole']:
            raise ValueError('Argument: type must be str:mass_spectrum or str:whole')

        if dim is None:
            raise RuntimeError('Argument: dim must be set if the GCMSReduceDimension is apply.')

        if not isinstance(dim, int):
            raise TypeError('Argument: dim must be a int.')

        if dilated_number is not None:
            if not isinstance(dilated_number, int):
                raise TypeError('Argument: dilated_number must be a int.')

        if mean_number is not None:
            if not isinstance(mean_number, int):
                raise TypeError('Argument: mean_number must be a int.')

        if stride is not None:
            if not isinstance(stride, int):
                raise TypeError('Argument: stride must be a int.')

        if dilated_number is not None and mean_number is not None:
            raise RuntimeError('Argument: dilated_number and mean_number can not be used simultaneously.')

        if not isinstance(retain_intensity, str):
            raise TypeError('Argument: retain_intensity must be a str.')

        if retain_intensity.lower() not in ['none', 'concat', 'multiply']:
            raise ValueError('Argument: retain_intensity only work if reduce the dim of the mass spectra, it must be str:none, str:concat or str:multiply.')

        if not isinstance(unit_vector, bool):
            raise TypeError('Argument: unit_vector must be a bool.')

        if not isinstance(batch, bool):
            raise TypeError('Argument: batch must be a bool.')

        if type.lower() == 'whole':
            logging.warning("Only argument: dim will work if the mode is 'whole'.")

        self.type = type.lower()
        self.dim = dim
        self.dilated_number = dilated_number
        self.mean_number = mean_number
        self.stride = stride
        self.retain_intensity = retain_intensity.lower()
        self.unit_vector = unit_vector
        self.batch = batch

        self.pca_model = None
        self.flatten = GCMSFlatten(batch = batch)
        self.eps = 1e-8

    def add_pca(self, pca):
        if not isinstance(pca, sklearn.decomposition.PCA):
            raise TypeError('Added PCA model must use sklearn.decomposition.PCA object.')

        self.pca_model = pca

        return None

    def __call__(self, spectra, return_type = 'origin'):
        if return_type.lower() not in ['origin', 'numpy', 'torch']:
            raise ValueError('Argument: return_type must be str:origin, str:numpy or str:torch')

        if isinstance(spectra, np.ndarray):
            dim = len(spectra.shape)
        elif isinstance(spectra, torch.Tensor):
            dim = len(spectra.size())
            spectra = spectra.cpu().numpy()
        else:
            raise TypeError('Input spectra must be numpy.ndarray or torch.Tnesor.')

        if self.type == 'whole':
            if self.batch:
                if dim == 3:
                    spectra = self.flatten(spectra, return_type = 'numpy')
                elif dim == 1 or dim > 3:
                    raise RuntimeError('Input spectra must be 2d (num_samples, feature) or 3d (num_sample, retention_time, m/z) array.')

                if self.unit_vector:
                    spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

                spectra = self.pca_model(spectra)
                
            else:
                if dim == 2:
                    spectra = self.flatten(spectra, return_type = 'numpy')
                elif dim > 2:
                    raise RuntimeError('Input spectra must be 2d (feature) or 3d (retention_time, m/z) array.')

                spectra = np.expand_dims(spectra, axis = 0)
                if self.unit_vector:
                    spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

                spectra = self.pca_model.transform(spectra)
                spectra = np.squeeze(spectra, axis = 0)

        elif self.type == 'mass_spectrum':
            sample_number = spectra.shape[0]
            if self.batch:
                if dim != 3:
                    raise RuntimeError('Input spectra must be 3d (num_sample, retention_time, m/z) array.')

                if self.retain_intensity == 'concat' or self.retain_intensity == 'multiply':
                    intensity = np.sum(spectra, axis = 2)
                    if self.dilated_number is not None and self.mean_number is None:
                        intensity = intensity[:, :: self.dilated_number]
                    elif self.dilated_number is None and self.mean_number is not None:
                        intensity = self._mean_spectrum_by_time(intensity, self.mean_number, self.stride)

                    if self.unit_vector:
                        intensity = intensity / (np.linalg.norm(intensity, axis = 1, keepdims = True) + self.eps)

                spectra = spectra.reshape(-1, spectra.shape[-1])
                if self.unit_vector:
                    spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

                spectra = self.pca_model.transform(spectra)
                spectra = spectra.reshape(sample_number, -1, self.dim)

                if self.dilated_number is not None and self.mean_number is None:
                    spectra = spectra[:, :: self.dilated_number]
                elif self.dilated_number is None and self.mean_number is not None:
                    spectra = self._mean_spectrum_by_time(spectra, self.mean_number, self.stride)

                if self.retain_intensity == 'concat':
                    intensity = np.expand_dims(intensity, axis = 2)
                    spectra = np.concatenate((spectra, intensity), axis = 2)
                elif self.retain_intensity == 'multiply':
                    spectra *= np.expand_dims(intensity, axis = 2)

            else:
                if dim != 2:
                    raise RuntimeError('Input spectra must be 2d (retention_time, m/z) array.')

                if self.retain_intensity == 'concat' or self.retain_intensity == 'multiply':
                    intensity = np.sum(spectra, axis = 1)
                    if self.dilated_number is not None and self.mean_number is None:
                        intensity = intensity[:: self.dilated_number]
                    elif self.dilated_number is None and self.mean_number is not None:
                        intensity = self._mean_spectrum_by_time(np.expand_dims(intensity, axis = 0),
                                self.mean_number, self.stride)
                        intensity = np.squeeze(intensity, axis = 0)

                    if self.unit_vector:
                        intensity = intensity / (np.linalg.norm(intensity, axis = 0, keepdims = True) + self.eps)

                spectra = spectra.reshape(-1, spectra.shape[-1])
                if self.unit_vector:
                    spectra = spectra / (np.linalg.norm(spectra, axis = 1, keepdims = True) + self.eps)

                spectra = self.pca_model.transform(spectra)
                spectra = spectra.reshape(-1, self.dim)

                if self.dilated_number is not None and self.mean_number is None:
                    spectra = spectra[:: self.dilated_number]
                elif self.dilated_number is None and self.mean_number is not None:
                    spectra = self._mean_spectrum_by_time(np.expand_dims(spectra, axis = 0),
                            self.mean_number, self.stride)
                    spectra = np.squeeze(spectra, axis = 0)                    

                if self.retain_intensity == 'concat':
                    intensity = np.expand_dims(intensity, axis = 1)
                    spectra = np.concatenate((spectra, intensity), axis = 1)
                elif self.retain_intensity == 'multiply':
                    spectra *= np.expand_dims(intensity, axis = 1)

        return spectra

    def _mean_spectrum_by_time(self, spectra, mean_number, stride):
        if stride is None:
            stride = mean_number

        spectra = torch.tensor(spectra).unsqueeze(dim = 1)
        numbers = torch.ones(spectra.size(), dtype = torch.double, device = spectra.device)
        layer = nn.Conv2d(1, 1, kernel_size = (mean_number, 1),
                bias = False,
                stride = (stride, 1),
                padding = (mean_number - 1, 0),
                padding_mode = 'same')

        layer.weight = torch.nn.Parameter(
                torch.ones(layer.weight.size(), dtype = torch.double, device = layer.weight.device))

        spectra = layer(spectra)
        numbers = layer(numbers)

        spectra = (spectra / numbers).detach().cpu()
        spectra = np.array(spectra.squeeze(dim = 1))

        return spectra

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\ntype={0},\ndim={1},\n'.format(self.type, self.dim)
        if self.dilated_number is not None:
            format_string += 'dilated_number={0},\n'.format(self.dilated_number)

        if self.mean_number is not None:
            format_string += 'mean_number={0},\n'.format(self.mean_number)

        if self.stride is not None:
            format_string += 'stride={0},\n'.format(self.stride)

        format_string += 'retain_intensity={0},\nunit_vector={1},\nbatch={2})'\
                .format(self.retain_intensity, self.unit_vector, self.batch)

        return format_string


