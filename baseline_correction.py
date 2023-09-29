import numpy as np
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