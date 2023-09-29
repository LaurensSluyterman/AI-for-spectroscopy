import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from airPLS import airPLS
from functools import partial
from baseline_correction import minimum_correction, zero_correction
from pipeline import Model, leave_one_out_error
from utils import resize, absorption_to_absorbance
from data_simulation import gen_data_set

#%% import data
# matplotlib.use('TkAgg')

# breath data
data_breath = pd.read_csv('PLS_data/PLS_Breath.txt', header=None)
start_2 = int(len(data_breath) / 5)  # we do not want the part 800-900
wave_numbers_breath = data_breath.to_numpy()[start_2:, 0]
absorption_spectra_breath = data_breath.to_numpy()[start_2:, 1:].transpose()
covariates_breath = absorption_to_absorbance(absorption_spectra_breath)

# calibration data
data_test = pd.read_csv('PLS_data/PLS_Calibration.txt', header=None)
wave_numbers_test = data_test.to_numpy()[start_2:, 0]
covariates_test = -  np.log(1-data_test.to_numpy()[start_2:, 1:].transpose())

# training data
data = pd.read_csv('PLS_data/PLS_X.txt', header=None)
absorption_spectra = data.to_numpy()[:, 1:].transpose()
concentrations_and_labels = pd.read_csv('PLS_data/PLS_Y.txt', header=None)
concentrations = concentrations_and_labels.to_numpy()[:, 1:].transpose()
labels = concentrations_and_labels.to_numpy()[:, 0].transpose()
wave_numbers = data.to_numpy()[:, 0]
absorption_spectra_resized = resize(wave_numbers_breath,
                                    original_wave_numbers=wave_numbers,
                                    original_covariates=absorption_spectra)

# Import background spectra and rescale to size of breath spectra
background_1 = scipy.io.loadmat('background-spectra/MO97_PA100_3GHz_17-05_breath_background_1.mat')
background_2 = scipy.io.loadmat('background-spectra/MO97_PA100_3GHz_31-05_background_1.mat')
background_3 = scipy.io.loadmat('background-spectra/MO97_PA100_3GHz_01-06_background.mat')
background_4 = scipy.io.loadmat('background-spectra/MO97_PA100_3GHz_17-05_breath_background_2.mat')
wave_numbers_background = scipy.io.loadmat("background-spectra/wavenumber_grid.mat")['wavenumber_grid'][0]
backgrounds = [background_1['spec_sig_full_reduced'].T,
               background_2['spec_sig_full_reduced'].T,
               background_3['spec_sig_full_reduced'].T,
               background_4['spec_sig_full_reduced'].T]
# Resize the backgrounds to the correct grid
for i, background in enumerate(backgrounds):
    backgrounds[i] = resize(wave_numbers_breath,
                            original_wave_numbers=wave_numbers_background,
                            original_covariates=background)

#%% Generate a training set using the absorption spectra and backgrounds
covariates, targets = gen_data_set(500, absorption_spectra_resized,
                                   concentrations, backgrounds)
X_train, X_test, Y_train, Y_test = train_test_split(covariates, targets,
                                                    test_size=0.1, random_state=2)

# Plot a newly simulated spectrum
plt.plot(wave_numbers_breath, X_train[1])
plt.show()

#%% Fit a PLS model (possibly with background-correction).
Model1 = Model(model=PLSRegression(n_components=50, scale=False),
               target_normalization=True,
               baseline_correction=partial(airPLS, itermax=100, lambda_=25))
Model1.fit(X_train, Y_train)

# Show performance. Right now a bit unfair since the training set does not have a good split.
Y_pred = (Model1.predict(X_test))
for i in range(11):
    plt.title(labels[i])
    plt.xlabel('true concentration')
    plt.ylabel('predicted concentration')
    plt.scatter(Y_test[:, i], Y_pred[:, i])
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
    name = labels[i]
    # plt.savefig('./plots/predictedvsobserved' + name)
    # plt.close()
    plt.show()

leave_one_out_error(Model1, X_train, Y_train)

#%% Test on breath data
plt.plot(wave_numbers_breath, covariates_breath[0])
plt.show()

Y_pred_breath = Model1.predict(covariates_breath)

for i, concentration in enumerate(Y_pred_breath[0]):
    if concentration < 0:
        print(f'{labels[i]}: {concentration * 1e9} ppb')
    elif concentration < 1e-4:
        print(f'{labels[i]}: {concentration * 1e9} ppb')
    else:
        print(f'{labels[i]}: {concentration * 1e2} %')

#%% Some examples of baseline_correction
plt.title('Baseline')
plt.plot(wave_numbers_breath, covariates_breath[0])
plt.plot(wave_numbers_breath, airPLS(covariates_breath[0], itermax=100, lambda_=25), linewidth=4)
plt.ylabel('a')
plt.xlabel('cm-1')
plt.show()

plt.title('Corrected')
plt.plot(wave_numbers_breath, covariates_breath[0] - airPLS(covariates_breath[0], itermax=100, lambda_=25))
plt.ylabel('a')
plt.xlabel('cm-1')
plt.show()

plt.title('zero_corrected')
plt.plot(wave_numbers_breath, covariates_breath[0] - zero_correction(covariates_breath[0]))
plt.ylabel('a')
plt.xlabel('cm-1')
plt.show()

plt.plot(wave_numbers_breath, X_train[1])
plt.plot(wave_numbers_breath, airPLS(X_train[1], itermax=100, lambda_=200))
plt.show()

plt.plot(wave_numbers_test, covariates_test[2])
plt.plot(wave_numbers_test, airPLS(covariates_test[2], lambda_=25))
plt.show()

#%% Minimum correction
plt.title('Simulated data')
plt.plot(wave_numbers_breath, covariates[0])
plt.ylabel('a')
plt.xlabel('cm-1')
plt.show()

plt.title('Minimum method')
plt.plot(wave_numbers_breath, covariates_breath[0])
plt.plot(wave_numbers_breath, minimum_correction(covariates_breath[0], 100))
plt.show()
