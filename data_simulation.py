import numpy as np
import random
from random import randint
from utils import absorption_to_absorbance
from sklearn.utils import resample


def resample_spectra(spectra_1, spectra_2=None, size=None):
    """
    Get a two bootstrapped background samples.

    This function uses bootstrapping to get a newly sampled set of spectra.
    The goal is to use a set of background spectra and to resample this to
    get a new realisation of the noise. Additionally, we do this in such a way
    to also have a background drift. This is done as follows.

    Two sets of spectra is given. From each set, we create a new set of
    spectra of either the same size or the specified size.

    The goal of these spectra is to use the pair to create a realistic
    new absorbance spectrum to train a model on. By sampling in this way,
    we incorporate both the random noise effects and the background drift.

    Arguments:
        spectra_1 (np.array): An array containing multiple subsequent measurements
            of a power spectrum. Each row corresponds to a different measurement.
        spectra_2: (np.array). An array containing multiple subsequent measurements
            of a power spectrum. Each row corresponds to a different measurement.
        size: The desired number of measurements of the sets of spectra. If not
            specified, the number will be the same as the number of measurements
            of the first spectrum.

    Returns:
        new_spectra_1: A newly sampled set of spectra
        new_spectra_2: A newly sampled set of spectra
    """
    if size is None:
        size = np.shape(spectra_1)[0]
        new_spectra_1 = resample(spectra_1, n_samples=size, replace=True)
        new_spectra_2 = resample(spectra_2, n_samples=size, replace=True)
    return np.array(new_spectra_1), np.array(new_spectra_2)


def apply_transmission(spectra_average, transmission):
    """Apply a transmission spectrum on an averaged background spectrum."""
    assert len(spectra_average) == len(transmission)
    return spectra_average * transmission


def get_absorption_spectrum(transmission_spectrum, spectra_1, spectra_2=None,
                            size=None):
    """
    Combine a transmission spectrum and background spectrum(s) into absorption.
    """
    # Resample
    new_spectra_1, new_spectra_2 = resample_spectra(spectra_1, spectra_2, size)

    # Apply transmission
    I_x = apply_transmission(np.mean(new_spectra_1, axis=0), transmission_spectrum)
    I_0 = np.mean(new_spectra_2, axis=0)
    absorption_spectrum = 1 - I_x / I_0
    return absorption_spectrum


def gen_data_set(N_samples, absorption_list, concentrations, background_list):
    """
    Create a simulated data set.

    This function creates N_samples absorbance spectra based on a list of
    absorption spectra and corresponding concentrations and a list of background
    intensity spectra.

    Two distinct background spectra are sampled at random. Subsequently, the
    get_absorption_spectrum is used to combine these two background spectra and
    the selected absorption spectrum to a new absorption spectrum. This
    spectrum contains background drift due to the combination of the two
    backgrounds and has other noise effect via bootstrapping.

    Arguments:
        N_samples (int): The number of generated absorbance spectra.
        absorption_list: A list containing multiple simulated absorption spectra.
        concentrations: A list containing the accompanying concentrations.
        background_list: A list containing multiple background spectra.

    Returns:
        covariates: An array where each row is a newly simulated absorbance
            spectrum.
        targets: The concentrations corresponding to the covariates.
    """
    covariates = np.zeros((N_samples, len(background_list[0][0])))
    targets = np.zeros((N_samples, len(concentrations[0])))
    for i in range(N_samples):
        # Randomly select one of the concentrations
        j = randint(0, len(concentrations) - 1)

        # Randomly select two background spectra
        indices = random.sample([i for i in range(len(background_list))], 2)
        spectra_1 = background_list[indices[0]]
        spectra_2 = background_list[indices[1]]

        # Combine the selected background spectra and absorption/concentration
        # to construct a new absorbance spectrum
        absorption = get_absorption_spectrum(1 - absorption_list[j],
                                             spectra_1=spectra_1,
                                             spectra_2=spectra_2)
        absorbance = absorption_to_absorbance(absorption)
        covariates[i] = absorbance
        targets[i] = concentrations[j]
    return covariates, targets
