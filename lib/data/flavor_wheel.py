import os
import logging

import numpy as np

from ..utils import save_object, load_json_obj

from .utils import init_directory

#--------------------------------------------------------------------------------
# flavor_wheel.py contains the flavor wheel object which can be used to display
# labels, and it can transformed the description labels into multi-hot labels based
# on the descriptions on the coffee flavor wheel
#--------------------------------------------------------------------------------


__all__ = ['FlavorWheel']


class FlavorWheel(object):
    def __init__(self, path):
        self.path = path

        self.save_path = init_directory(path)
        self.wheel = self.build_wheel(self.save_path)
        self.labels = self.build_labels(self.save_path)

    def save_all(self):
        save_object(os.path.join(self.save_path, 'flavor_wheel'), self.wheel, 'json')

        for layer in self.labels:
            save_object(os.path.join(self.save_path, layer + '_label'), self.labels[layer], 'json')

        logging.info('All object saving done.')
        return None

    def label_transform_matrix(self, descriptions, layer):
        wheel_descriptions = self.labels[layer]
        description_keys = list(wheel_descriptions.keys())

        transform_matrix = np.zeros((len(descriptions), len(wheel_descriptions)))
        for row_index in range(len(descriptions)):
            for column_index in range(len(description_keys)):
                if descriptions[row_index] in self.labels[layer][description_keys[column_index]]:
                    transform_matrix[row_index, column_index] = 1

        transform_matrix = transform_matrix.astype(np.int)

        return transform_matrix, wheel_descriptions

    def build_wheel(self, path):
        filename = os.path.join(path, 'flavor_wheel.json')
        if os.path.isfile(filename):
            wheel = load_json_obj(filename)
        else:
            wheel = self.build_new_wheel()

        return wheel

    def build_labels(self, path):
        labels = {}
        for layer in self.wheel:
            filename = os.path.join(path, layer + '_label.json')
            if os.path.isfile(filename):
                file_label = load_json_obj(filename)

                labels[layer] = {}
                categories = self.wheel[layer]
                for category in categories:
                    labels[layer][category] = file_label[category]
            else:
                raise RuntimeError('Please use ./unix_scripts/update_flavor_wheel.sh to construct the flavor wheel.')

        return labels


