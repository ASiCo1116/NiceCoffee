import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from ..utils import save_object, load_pickle_obj

from .utils import EmptyFunction
from .base import DeepLearningModule

#--------------------------------------------------------------------------------
# binary_mask_estimator.py contain the binary mask estimator for using gumbel softmax
# reparametric trick.
#--------------------------------------------------------------------------------


__all__ = ['BinaryMaskEstimatingModule']


class BinaryMaskEstimatingModule(DeepLearningModule):
    def __init__(self,
            mask_shape = (),
            mapping_function = EmptyFunction,
            mapping_shape = None,
            estimated_model = None,
            loss_function = nn.L1Loss(),
            init_energy_shifting = 1.,
            init_energy_scaling = 1e-3,
            output_mask_energy_scaling = 10.,
            gp_lambda = 1.,
            temperature = 1e-20,
            eps = 1e-20):

        super(DeepLearningModule, self).__init__()

        if not isinstance(mask_shape, (list, tuple)):
            raise TypeError('Argument: mask_shape must be a list or tuple.')

        if sum(mask_shape) <= 0:
            raise RuntimeError('Argument: mask shape must at least have one number.')

        if mapping_function is not None:
            if not callable(mapping_function):
                raise RuntimeError('Argument: mapping_function must be a callable function or object.')

        if estimated_model is not None:
            if not isinstance(estimated_model, nn.Module):
                raise TypeError('Argument: estimated_model must be the object inherited from torch.nn.Module.')

        if loss_function is not None:
            if not callable(loss_function):
                raise TypeError('Argument: loss_function must be a callable object.')

        if not isinstance(init_energy_shifting, float):
            raise TypeError('Argument: init_energy_shit must be a float.')

        if not isinstance(init_energy_scaling, float):
            raise TypeError('Argument: init_energy_scaling must be a flaot.')

        if not isinstance(output_mask_energy_scaling, float):
            raise TypeError('Argument: output_mask_energy_scaling must be a float.')

        if not isinstance(gp_lambda, float):
            raise TypeError('Argument: gp_lambda must be a float.')

        if not isinstance(temperature, float):
            raise TypeError('Argument: temperature must be a float.')

        if not isinstance(eps, float):
            raise TypeError('Argument: eps was recommended being a small number near zero and type is float.')

        self.mask_shape = mask_shape

        self.mapping_function = mapping_function
        self.estimated_model = estimated_model
        self.loss_function = loss_function

        self.init_energy_shifting = init_energy_shifting
        self.init_energy_scaling = init_energy_scaling
        self.output_mask_energy_scaling = output_mask_energy_scaling
        self.gp_lambda = gp_lambda

        self.temperature = temperature
        self.eps = eps

        # check mapping function
        t = torch.randn(mask_shape)
        t = self.mapping_function(t)
        if mapping_shape is None:
            self.mapping_shape = t.size()
        else:
            if mapping_shape != t.size():
                raise RuntimeError('Size mismatch: the shape of mapped mask is not equal to input shape.')

            self.mapping_shape = mapping_shape

        self.mask_slice = [slice(None, None, None) for i in range(len(mask_shape))]
        self.mask_slice.append(0)
        self.mask_slice = tuple(self.mask_slice)
        self.mapping_mask_slice = [slice(None, None, None) for i in range(len(self.mapping_shape))]
        self.mapping_mask_slice.append(0)
        self.mapping_mask_slice = tuple(self.mapping_mask_slice)

        another_channel = [slice(None, None, None) for i in range(len(self.mapping_shape))]
        another_channel.append(1)
        another_channel = tuple(another_channel)

        # double channel parameter initialization
        energy_shape = list(mask_shape)
        energy_shape.append(2)

        self.energy = torch.randn(energy_shape) * init_energy_scaling
        self.energy[self.mask_slice] -= init_energy_shifting
        self.energy[another_channel] += init_energy_shifting
        self.energy = nn.Parameter(self.energy)

        self.estimation_finish = False

    def mask(self, sample_times = 20, threshold = 0.99):
        return self.catch_mask(sample_times = sample_times, threshold = threshold)

    def mapping_mask(self, sample_times = 20, threshold = 0.99):
        return self.catch_mask(sample_times = sample_times, threshold = threshold)

    def filtered_mask(self, ratio, sample_times = 20, threshold = 0.5):
        return self.catch_filtered_mask(ratio = ratio, sample_times = sample_times, threshold = threshold)

    def filtered_mapping_mask(self, ratio, sample_times = 20, threshold = 0.5):
        return self.catch_filtered_mapping_mask(ratio = ratio, sample_times = sample_times, threshold = threshold)
    
    def catch_energy(self):
        energy = self.energy.clone().detach()
        return energy

    def catch_mask_energy(self):
        energy = self.energy.clone().detach()[self.mask_slice]
        return energy

    def catch_mapping_mask_energy(self):
        energy = self.energy.clone().detach()
        if self.mapping_function is not None:
            energy = self.mapping_function(energy)

        return energy[self.mapping_mask_slice]

    def catch_filtered_energy(self, ratio):
        if not isinstance(ratio, float):
            raise TypeError('Argument: ratio must be a float.')

        if ratio > 1 or ratio <= 0.:
            raise ValueError('Argument: ratio must bigger than zero, and maximum value is one.')

        filtered_energy = self.energy.clone().detach()
        first_channel = filtered_energy[self.mask_slice].clone()
        best_k = math.ceil(float(first_channel.numel()) * ratio)
        values, _ = torch.topk(first_channel.view(-1), k = best_k)
        k_large = float(values[-1])
        mask = (first_channel > k_large).float().unsqueeze(len(first_channel.size()))
        filtered_energy = filtered_energy * mask 

        return filtered_energy.detach()

    def catch_mask(self, sample_times = 20, threshold = 0.9):
        energy = self.energy.clone()
        energy *= self.output_mask_energy_scaling

        mask = [self.st_gumbel_softmax_estimator(
            energy = energy,
            mapping_function = EmptyFunction)
                for i in range(sample_times)]

        mask = torch.round(torch.stack(mask, dim = 0).mean(dim = 0))
        mask = mask * (mask > threshold).float()

        return mask.detach().byte()

    def catch_mapping_mask(self, sample_times = 20, threshold = 0.9):
        energy = self.energy.clone()
        energy *= self.output_mask_energy_scaling
        
        mask = [self.mapping_function(self.st_gumbel_softmax_estimator(
            energy = energy)) for i in range(sample_times)]

        mask = torch.round(torch.stack(mask, dim = 0).mean(dim = 0))
        mask = mask * (mask > threshold).float()

        return mask.detach().byte()

    def catch_filtered_mask(self, ratio, sample_times = 20, threshold = 0.9):
        filtered_energy = self.catch_filtered_energy(ratio = ratio)
        filtered_energy *= self.output_mask_energy_scaling

        filtered_mask = [self.st_gumbel_softmax_estimator(
            energy = filtered_energy,
            mapping_function = EmptyFunction)
                for i in range(sample_times)]

        filtered_mask = torch.round(torch.stack(filtered_mask, dim = 0).mean(dim = 0))
        filtered_mask = filtered_mask * (filtered_mask > threshold).float()

        return filtered_mask.detach().byte()

    def catch_filtered_mapping_mask(self, ratio, sample_times = 20, threshold = 0.9):
        filtered_energy = self.catch_filtered_energy(ratio = ratio)
        filtered_energy *= self.output_mask_energy_scaling

        filtered_mask = [self.st_gumbel_softmax_estimator(
            energy = filtered_energy)
            for i in range(sample_times)]

        filtered_mask = torch.round(torch.stack(filtered_mask, dim = 0).mean(dim = 0))
        filtered_mask = filtered_mask * (filtered_mask > threshold).float()

        return filtered_mask.detach().byte()

    def save(self, path):
        cpu_model = self.cpu()

        state = {}
        state['model_type'] = 'BinaryMaskEstimatingModule'
        state['mask_shape'] = self.mask_shape
        state['mapping_function'] = self.mapping_function
        state['mapping_shape'] = self.mapping_shape
        state['estimated_model'] = self.model
        state['loss_function'] = self.loss_function
        state['temperature'] = self.temperature
        state['eps'] = self.eps
        state['estimation_finish'] = self.estimation_finish
        state['_iter_index'] = self._iter_index

        state['state_dict'] = cpu_model.state_dict()

        save_object(path, state, mode = 'pickle', extension = 'bme')
        logging.info('Model: BinaryMaskEstimatingModule saving done.')

        return None

    def load(self, path):
        if not fname.endswith('.bme'):
           raise RuntimeError('lib.model.BinaryMaskEstimatingModule only can load file endswith .bme which is saved by this object.')

        state = load_pickle_obj(fname, extension_check = False)

        if state['model_type'] != 'BinaryMaskEstimatingModule':
            raise RuntimeError('The object is not a state file which is trained by this object.')

        self.__init__(
                mask_shape = state['mask_shape'],
                mapping_function = state['mapping_function'],
                mapping_shape = state['mapping_shape'],
                estimated_model = state['estimated_model'],
                loss_function = state['loss_function'],
                temperature = state['temperature'],
                eps = state['eps'])

        self.estimation_finish = state['estimation_finish']
        self._iter_index = state['_iter_index']

        self.load_state_dict(state['state_dict'])

        logging.info('Load state file success !!')

        return self

    def model_output(self, data):
        device, dtype = data.device, data.dtype
        model = self.estimated_model.to(device)
        return model(data)

    def forward(self,
            data,
            label = None,
            label_grad_filter = None,
            model = None,
            loss_function = None,
            temperature = None,
            gp_lambda = None,
            train = False):

        if model is None:
            model = self.estimated_model

        if loss_function is None:
            loss_function = self.loss_function

        if temperature is None:
            temperature = self.temperature

        device, dtype = data.device, data.dtype
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        if isinstance(loss_function, nn.Module):
            loss_function = loss_function.to(device)

        mask = self.st_gumbel_softmax_estimator(temperature = temperature).type(dtype)

        masked_data = data.clone() * mask
        original_output = model(data)
        original_output = original_output.detach()
        masked_output = model(masked_data)

        if label is None:
            label = original_output

        if label_grad_filter is not None:
            label *= label_grad_filter

        if gp_lambda is None:
            gp_lambda = self.gp_lambda

        if gp_lambda != 0:
            loss_value = loss_function(masked_output, label) + gp_lambda * self._gradient_penalty(model, data, masked_data)
        else:
            loss_value = loss_function(masked_output, label) 

        return loss_value

    def _gradient_penalty(self, model, data, masked_data):
        shape = [1 for i in range(len(data.size()) - 1)]
        shape.insert(0, data.size(0))
        alpha = torch.rand(shape, dtype = data.dtype, device = data.device)
        interpolates = (alpha * data + (1 - alpha) * masked_data).requires_grad_(True)
        model_interpolates = model(interpolates)

        mapping_dist = torch.ones(model_interpolates.size(),
                dtype = model_interpolates.dtype, device = model_interpolates.device).requires_grad_(False)

        gradients = torch.autograd.grad(
                outputs = model_interpolates,
                inputs = interpolates,
                grad_outputs = mapping_dist,
                create_graph = True,
                retain_graph = True,
                only_inputs = True
                )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(p = 2, dim = 1) - 1) ** 2).mean()

        return gradient_penalty

    # ref: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
    def st_gumbel_softmax_estimator(self,
            energy = None,
            mapping_function = None,
            temperature = None):

        if temperature is None:
            temperature = self.temperature

        if energy is None:
            if mapping_function is None:
                energy = self.mapping_function(self.energy)
            else:
                energy = mapping_function(self.energy)
        else:
            if not isinstance(energy, torch.Tensor):
                raise TypeError('The input of the ST-Gumbel-softmax estimator must be a torch.Tensor.')

        device, shape = energy.device, energy.size()
        distribution = self._gumbel_softmax_sampling(energy, temperature, device = device)
        _, indices = distribution.max(dim = -1)
        hard_dist = torch.zeros_like(distribution).view(-1, shape[-1])
        hard_dist.scatter_(1, indices.view(-1, 1), 1)
        hard_dist = hard_dist.view(*shape)

        categorical = (hard_dist - distribution).detach() + distribution

        return categorical[self.mask_slice]

    def _gumbel_softmax_sampling(self, energy, temperature, device = None):
        if device is None:
            device = torch.device('cpu')

        distribution = (energy + self._gumbel_sampling(energy.size(), device = device)) / temperature
        distribution = torch.softmax(distribution, dim = -1)

        return distribution

    def _gumbel_sampling(self, shape, device, eps = None):
        if eps is None:
            eps = self.eps

        U = torch.rand(shape, requires_grad = True)
        distribution = -torch.log(-torch.log(U + eps) + eps)

        return distribution.to(device)


