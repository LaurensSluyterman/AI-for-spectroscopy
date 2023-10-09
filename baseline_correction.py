import numpy as np
import scipy
from copy import deepcopy

def minimum_correction(spectrum, width=100):
    baseline = deepcopy(spectrum)
    for i, value in enumerate(spectrum):
        min_index = max(0, i - width)
        max_index = min(len(spectrum), i+width)
        relevant_values = []
        for j in range(min_index, max_index):
            relevant_values.append(spectrum[j])
        offset = np.min(relevant_values)
        baseline[i] = offset
    return baseline


def zero_correction(spectrum):
    baseline = deepcopy(spectrum)
    for i, value in enumerate(spectrum):
        baseline[i] = np.min((0, value))
    return baseline

def SG_filter(spectrum, window_length, poly_order, deriv_order):
    a = scipy.signal.savgol_filter(spectrum, window_length, poly_order, deriv=deriv_order, delta=1.0, axis=-1, mode='interp', cval=0.0)
    baseline = spectrum - a
    return baseline