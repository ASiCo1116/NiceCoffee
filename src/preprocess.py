import numpy as np

from scipy.signal import savgol_filter
#--------------------------------------------------------------------------------
#the preprocess.py is used in the flavor model of the coffee recognition
#the preprocess.py contain all the function used in the project of coffee NIR
#--------------------------------------------------------------------------------

__all__ = ['Preprocess', 'MSC', 'SNV', 'SG']

def Preprocess(data, preprocessing, path = '../object'):
    if preprocessing == 'MSC' or preprocessing == 'msc':
        data, _ref = MSC(data, reference = np.load(path + '/MSC_ref.npy'))
    elif preprocessing == 'SNV' or preprocessing == 'snv':
        data = SNV(data)
    elif 'SG' in preprocessing or 'sg' in preprocessing:
        data = SG(data, preprocessing)
    else:
        pass

    return data

def MSC(input_spectrum, reference = None):
    #the function is the correction of the NIR data
    #input data is a samples * one dimension  feature data

    #mean central correction
    for i in range(input_spectrum.shape[0]):
        input_spectrum[i, :] -= input_spectrum[i, :].mean()

    #define the reference of msc
    if reference is None:
        ref = np.mean(input_spectrum, axis = 0)
    else:
        ref = reference

    #define a new array for the data after correction
    msc_spectrum = np.zeros_like(input_spectrum)
    for i in range(input_spectrum.shape[0]):
        #to run the regression for correction
        fit = np.polyfit(ref, input_spectrum[i, :], 1, full = True)
        #apply correction
        msc_spectrum[i, :] = (input_spectrum[i, :] - fit[0][1]) / fit[0][0]

    return (msc_spectrum, ref)

def SNV(input_spectrum):
    #the function is the correction of the NIR data
    #input data is a samples * one dimension  feature data

    #define a new array for the data after correction
    snv_spectrum = np.zeros_like(input_spectrum)
    for i in range(input_spectrum.shape[0]):
        snv_spectrum[i, :] = (input_spectrum[i, :] - np.mean(input_spectrum[i, :])) / np.std(input_spectrum[i, :])

    return snv_spectrum

def SG(input_spectrum, arguments):
    #the function is the correction of the NIR data
    #input data is a samples * one dimension  feature data
    #savitzky golay filter
    def _decode_argument(arguments):
        arguments = arguments.split('_')
        poly, der, win = 0, 0, 1
        for obj in arguments:
            if 'help' in obj:
                print('There were three arguments can be set in SG filter, [poly_order, derivative order, window length].')
                print('All of the arguments must be a int. Here is example: SG_w5_p2_d2')
                exit(0)

            else:
                if 'p' in obj:
                    poly = int(obj.replace('p', ''))
                elif 'd' in obj:
                    der = int(obj.replace('d', ''))
                elif 'w' in obj:
                    temp = int(obj.replace('w', ''))
                    if temp % 2 == 0 or temp < 0:
                        raise ValueError('window length of SG filter must be a positive and odd number.')
                    else:
                        win = temp

        if poly >= win:
            raise ValueError('polyorder must be less than window_length.')

        if der < 0:
            raise ValueError('This must be a nonnegative integer.')

        return poly, der, win

    poly, der, win = _decode_argument(arguments)

    for i in range(input_spectrum.shape[0]):
        input_spectrum[i] = savgol_filter(input_spectrum[i], window_length = win, polyorder = poly, deriv = der)

    if win == 1:
        adjust_length = 0
    else:
        adjust_length = win // 2

    if adjust_length > 0:
        input_spectrum[:, -adjust_length:] = 0.0
        input_spectrum[:, 0: adjust_length] = 0.0

    return input_spectrum