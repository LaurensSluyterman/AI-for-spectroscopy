import numpy as np
import scipy
from copy import deepcopy

def minimum_correction(spectrum, width=100):
    """
    Get a baseline using minimum method.

    This function takes a spectrum and returns
    a baseline. For position i, the baseline is defined as the minimum value of
    the interval [i - width, i + width]. If the interval exceeds the spectrum,
    it is cut of.

    Arguments:
        spectrum: The spectrum.
        width (int): The length values that should be taken into account at
            either side to determine the minimum value. Larger values results
            in a smoother baseline.

    Returns:
        baseline: The baseline
    """
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
    """Set negative values to zero."""
    baseline = deepcopy(spectrum)
    for i, value in enumerate(spectrum):
        baseline[i] = np.min((0, value))
    return baseline

def SG_filter(spectrum, window_length, poly_order, deriv_order):
    """
    Apply SG filter.

    This function applies an SG filter. See the scipy documentation for full
    explanation of the arguments. The baseline is defined as the difference
    between the spectrum and the filtered spectrum. This is done in this way
    since it can be used by other code as baseline-correction with as effect
    that the spectrum gets filtered.
    """
    a = scipy.signal.savgol_filter(spectrum, window_length, poly_order, deriv=deriv_order, delta=1.0, axis=-1, mode='interp', cval=0.0)
    baseline = spectrum - a
    return baseline