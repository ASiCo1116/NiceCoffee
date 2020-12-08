import math
import random
import logging

import numpy as np

import torch


__all__ = ['BaseSingleRatioTuner', 'MultiClassSingleRatioTuner', 'MultiLabelSingleRatioTuner']


class BaseSingleRatioTuner(object):
    def __init__(self,
            lower_bound = 0.2,
            min_select_ratio = 0.5):

        self.lower_bound = self._ratio_check(lower_bound)
        self.min_select_ratio = self._ratio_check(min_select_ratio)

    def _ratio_check(self, ratio):
        if not isinstance(ratio, float):
            raise TypeError('Argument: ratio must be a float.')

        if ratio < 0 or ratio > 1:
            raise ValueError('Argument: ratio must at least 0 and can not bigger than 1.')

        return ratio

    def set_lower_bound_ratio(self, ratio):
        self.lower_bound = self._ratio_check(ratio)

        return self

    def set_select_ratio(self, ratio):
        self.min_select_ratio = self._ratio_check(ratio)

        return self


class MultiClassSingleRatioTuner(BaseSingleRatioTuner):
    def __init__(self,
            n_classes,
            lower_bound = 0.2,
            min_select_ratio = 0.5):

        super(MultiClassSingleRatioTuner, self).__init__(
                lower_bound = lower_bound,
                min_select_ratio = min_select_ratio)

        if not isinstance(n_classes, int):
            raise TypeError('Argument: n_classes must be a int.')

        if n_classes <= 0:
            raise ValueError('Argument: n_classes must bigger than zero.')

        self.n_classes = n_classes

    def __repr__(self):
        text = self.__class__.__name__ + '(n_classes={0}, lower_bound={1}, min_select_ratio={2})'\
                .format(self.n_classes, self.lower_bound, self.min_select_ratio)
        text += 'The ratio tuner API is designed for balancing the label distribution of the training process.\n'
        text += 'Note that some data will not be used in training when the ratio tuner was applied.'

        return text

    def tune(self, data, label, one_hot = False):
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError('Input data must be np.ndarray or torch.Tensor.')

        if not isinstance(label, (np.ndarray, torch.Tensor)):
            raise TypeError('Input label must be np.ndarray or torch.Tensor.')

        if not isinstance(one_hot, bool):
            raise TypeError('Argument: one_hot must be a bool.')

        if type(data) != type(label):
            raise RuntimeError('Input data and label must be in the sample type.')

        if len(data) != len(label):
            raise RuntimeError('Size mismatch error: input data and label do not has the same length.')

        if len(label) <= 0 or len(data) <= 0:
            raise RuntimeError('Length error: the length of input data must bigger than zero.')

        index_dict = self._statistic_label(label, one_hot = one_hot)
        new_classes_numbers = self._cal_class_number(
                index_dict,
                self.n_classes,
                self.lower_bound,
                self.min_select_ratio)

        new_index = self._resemble_index(index_dict, new_classes_numbers)
        new_data, new_label = [], []
        for index in new_index:
            new_data.append(data[index])
            new_label.append(label[index])

        if isinstance(label, np.ndarray):
            new_data = np.stack(new_data, axis = 0)
            new_label = np.stack(new_label, axis = 0)
            if len(label.shape) == 1:
                new_label = np.squeeze(new_label)

        elif isinstance(label, torch.Tensor):
            new_data = torch.stack(new_data, dim = 0)
            new_label = torch.stack(new_label, dim = 0)
            if len(label.size()) == 1:
                new_label = torch.squeeze(new_label)

            new_data = new_data.to(data.device)
            new_label = new_label.to(label.device)

        return new_data, new_label

    def _statistic_label(self, label, one_hot):
        index_dict = {}
        for i in range(len(label)):
            if label[i] not in index_dict.keys():
                index_dict[label[i]] = [i]
            else:
                index_dict[label[i]].append(i)

        return index_dict

    def _cal_class_number(self,
            index_dict,
            n_classes,
            lower_bound,
            min_select_ratio):

        lower_bound = (1. / n_classes) * (1. - lower_bound)

        keys = list(index_dict.keys())
        keys.sort()

        class_num = []
        for i in range(len(keys)):
            class_num.append(len(index_dict[keys[i]]))

        class_num = np.array(class_num)
        class_index = np.argsort(class_num)
        total_num = class_num.sum()
        remain_class_num = (np.sort(class_num)).astype(np.int)
        final_class_num = np.zeros((len(class_num)), dtype = np.int)
        min_select_num = np.ceil(min_select_ratio * total_num)

        if min_select_num >= total_num:
            final_class_num = remain_class_num.copy()
        else:
            needed_num = np.inf

            for i in range(len(final_class_num)):
                final_class_num, remain_class_num, needed_num = self._recursive_index_adding(
                        final_class_num,
                        remain_class_num,
                        needed_num,
                        i,
                        lower_bound,
                        min_select_num,
                        )

            finaL_class_num = self._final_index_adding(final_class_num, remain_class_num, needed_num)

        final_numbers = {}
        for i in range(len(keys)):
            final_numbers[keys[i]] = final_class_num[class_index[i]]

        return final_numbers

    def _recursive_index_adding(self,
            class_num,
            remain_num,
            needed_num,
            index,
            lower_bound,
            min_select_num):

        if needed_num == np.inf:
            least = np.min(remain_num)
            added_num = np.ones(self.n_classes) * least
        else:
            least = np.floor(needed_num / (self.n_classes - index))
            if least > np.min(remain_num):
                least = np.min(remain_num)

            added_num = np.ones(self.n_classes) * least

        added_num[: index] = 0
        added_num = added_num.astype(np.int)
        class_num += added_num
        remain_num -= added_num

        if needed_num == np.inf:
            needed_num = np.floor((least - lower_bound * least * self.n_classes) / lower_bound)
            if (needed_num + added_num.sum()) < min_select_num:
                needed_num = min_select_num - added_num.sum()
        else:
            needed_num -= np.sum(added_num)

        return class_num, remain_num, needed_num

    def _final_index_adding(self, class_num, remain_num, needed_num):
        needed_num = int(needed_num)
        iter_index = self.n_classes - 1
        last_added = np.zeros((self.n_classes), dtype = np.int)
        while needed_num > 0:
            if remain_num.sum() <= 0:
                break

            if remain_num[iter_index] > 0:
                last_added[iter_index] += 1
                remain_num[iter_index] -= 1
                needed_num -= 1
                iter_index -= 1
            else:
                iter_index = self.n_classes - 1

        return class_num + last_added 

    def _resemble_index(self, index_dict, class_numbers):
        selected_index = []
        for key in index_dict:
            class_index = index_dict[key]
            if len(class_index) == class_numbers[key]:
                selected = class_index.copy()
            else:
                selected = list(random.sample(class_index, class_numbers[key]))

            selected_index += selected

        return selected_index


class MultiLabelSingleRatioTuner(BaseSingleRatioTuner):
    def __init__(self,
            lower_bound = 0.2,
            min_select_ratio = 0.5):

        super(MultiLabelSingleRatioTuner, self).__init__(
                lower_bound = lower_bound,
                min_select_ratio = min_select_ratio)

        pass


