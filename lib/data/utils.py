import os

import numpy as np

#--------------------------------------------------------------------------------
# utils.py contains the function and object used by data object
#--------------------------------------------------------------------------------


__all__ = ['NIRExcelInfo', 'init_directory', 'select_nir_label_list', 'flavor_label_transform_matrix']


# New index for standard NIR excel files
NIRExcelInfo = {'Sample Number': [4, 5],
        'Label': [5, 6],
        'Agtron_Whole Bean': [6, 7],
        'Agtron_Ground': [7, 8],
        'CM_100w': [8, 9],
        'CM_100g': [9, 10],
        'Floral': [10, 11],
        'Tea-like': [11, 12],
        'Tropical_fruit': [12, 13],
        'Stone_fruit': [13, 14],
        'Citrus_fruit': [14, 15],
        'Berry_fruit': [15, 16],
        'Other_fruit': [16, 17],
        'Sour': [17, 18],
        'Alcohol': [18, 19],
        'Fermented': [19, 20],
        'Fresh Vegetable': [20, 21],
        'Dry Vegetable': [21, 22],
        'Papery/Musty': [22, 23],
        'Chemical': [23, 24],
        'Burnt': [24, 25],
        'Cereal': [25, 26],
        'Spices': [26, 27],
        'Nutty': [27, 28],
        'Cocoa': [28, 29],
        'Sweet': [29, 30],
        'Butter/Milky': [30, 31],
        'Baked': [31, 32],
        'Inner Floral': [10, 12],
        'Inner Fruity': [12, 17],
        'Inner Sour/Fermented': [17, 20],
        'Inner Green/Vegetable': [20, 22],
        'Inner Other': [22, 24],
        'Inner Roasted': [24, 26],
        'Inner Spices': [26, 27],
        'Inner Nutty/Cocca': [27, 29],
        'Inner Sweet': [29, 31],
        'Spectrum': [32, 932],
        'Index': None}

def init_directory(path):
    assert isinstance(path, str)

    if not os.path.isdir(path):
        os.makedirs(path)

    return path

def select_nir_label_list(select):
    if select.lower() == 'itri_inner':
        label_list = ['Inner Floral', 'Inner Fruity', 'Inner Sour/Fermented', 'Inner Green/Vegetable',
                'Inner Other', 'Inner Roasted', 'Inner Spices', 'Inner Nutty/Cocca', 'Inner Sweet', 
                'Baked', 'Index']

        return label_list

    elif select.lower() == 'inner':
        label_list = ['Inner Floral', 'Inner Fruity', 'Inner Sour/Fermented', 'Inner Green/Vegetable',
                'Inner Other', 'Inner Roasted', 'Inner Spices', 'Inner Nutty/Cocca', 'Inner Sweet',
                'Index']

        return label_list

    elif select.lower() == 'itri':
        label_list = ['Floral', 'Tea-like', 'Tropical_fruit', 'Stone_fruit', 'Citrus_fruit', 'Berry_fruit', 
            'Other_fruit', 'Sour', 'Alcohol', 'Fermented', 'Fresh Vegetable', 'Dry Vegetable', 'Papery/Musty',
            'Chemical', 'Burnt', 'Cereal', 'Spices', 'Nutty', 'Cocoa', 'Sweet', 'Butter/Milky', 'Index']

        return label_list

    elif select.lower() == 'itri_baked':
        label_list = ['Floral', 'Tea-like', 'Tropical_fruit', 'Stone_fruit', 'Citrus_fruit', 'Berry_fruit',
            'Other_fruit', 'Sour', 'Alcohol', 'Fermented', 'Fresh Vegetable', 'Dry Vegetable', 'Papery/Musty',
            'Chemical', 'Burnt', 'Cereal', 'Spices', 'Nutty', 'Cocoa', 'Sweet', 'Butter/Milky', 'Index']

        return label_list

    elif select.lower() == 'agtron':
        label_list = ['Index']

        return label_list

    else:
        raise ValueError('Only four mode (ITRI_inner, inner, ITRI, ITRI_baked) is available now.')

def flavor_label_transform_matrix(old_label, new_label):
    if old_label.lower() == 'itri' and new_label.lower() == 'inner':
        matrix = np.zeros((21, 9)).astype(np.int)
        matrix[0: 2, 0] = 1 # floral
        matrix[2: 7, 1] = 1 # fruity
        matrix[7: 10, 2] = 1 # sour/fermented
        matrix[10: 12, 3] = 1 # green/vegetable
        matrix[12: 14, 4] = 1 # other
        matrix[14: 16, 5] = 1 # roasted
        matrix[16: 17, 6] = 1 # spices
        matrix[17: 19, 7] = 1 # nutty/cocoa
        matrix[19: 21, 8] = 1 # sweet

    elif old_label.lower() == 'itri_baked' and new_label.lower() == 'inner_baked':
        matrix = np.zeros((22, 10)).astype(np.int)
        matrix[0: 2, 0] = 1 # floral
        matrix[2: 7, 1] = 1 # fruity
        matrix[7: 10, 2] = 1 # sour/fermented
        matrix[10: 12, 3] = 1 # green/vegetable
        matrix[12: 14, 4] = 1 # other
        matrix[14: 16, 5] = 1 # roasted
        matrix[16: 17, 6] = 1 # spices
        matrix[17: 19, 7] = 1 # nutty/cocoa
        matrix[19: 21, 8] = 1 # sweet
        matrix[21: 22, 9] = 1 # baked

    else:
        raise ValueError('The function not support ', old_label, ' transform into ', new_label)

    return matrix


