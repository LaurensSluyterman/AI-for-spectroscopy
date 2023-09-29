import numpy as np


def resize(new_wave_numbers, original_wave_numbers, original_covariates):
    """
    Resize covariates to a new grid.

    The wave number grid of different spectra is not always exactly equal.
    This function uses interpolation to convert covariates on a certain grid
    to covariates on a new grid.

    Arguments:
        new_wave_numbers (np.array): The desired grid.
        original_wave_numbers (np.array): The original grid.
        original_covariates (np.array): An array containing the original covariates
            each row is an entire spectrum. Each column corresponds to a single
            wave number.
    """
    new_covariates = np.zeros((len(original_covariates), len(new_wave_numbers)))
    for i, x in enumerate(original_covariates):
        new_covariates[i] = np.interp(new_wave_numbers, original_wave_numbers, x)
    return new_covariates


def absorption_to_absorbance(absorption_spectrum):
    """Convert absorption to absorbance."""
    absorbance_spectrum = - np.log(1 - absorption_spectrum)
    return absorbance_spectrum
