import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from functools import partial
from baseline_correction import SG_filter
from pipeline import Model
from utils import resize
from data_simulation import gen_data_set
from plot_functions import concentrations_plot


small_grid = np.loadtxt('PLS_data/small_grid.txt')

# training data
data = pd.read_csv('PLS_data/PLS_X2.txt', header=None)
absorption_spectra = data.to_numpy()[:, 1:].transpose()
concentrations_and_labels = pd.read_csv('PLS_data/PLS_Y2.txt', header=None)
concentrations = concentrations_and_labels.to_numpy()[:, 1:].transpose()
labels = concentrations_and_labels.to_numpy()[:, 0].transpose()
wave_numbers = data.to_numpy()[:, 0]
absorption_spectra_resized = resize(small_grid,
                                    original_wave_numbers=wave_numbers,
                                    original_covariates=absorption_spectra)

# Import background spectra and rescale to size of breath spectra
directory = './background_spectra_2'
background_names = file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
backgrounds = [scipy.io.loadmat(f'./{directory}/{name}')['spec_sig_full_reduced'].T for name in background_names]
wave_numbers_background = scipy.io.loadmat("background-spectra/wavenumber_grid.mat")['wavenumber_grid'][0]


# Resize the backgrounds to the correct grid
for i, background in enumerate(backgrounds):
    backgrounds[i] = resize(small_grid,
                            original_wave_numbers=wave_numbers_background,
                            original_covariates=background)


backgrounds_train, backgrounds_test = train_test_split(backgrounds, test_size=0.4)
absorptions_train, absorptions_test, concentrations_train, concentrations_test = train_test_split(absorption_spectra_resized,
                                                                                                  concentrations,
                                                                                                  random_state=2,
                                                                                                  test_size=0.2)
X_train, Y_train = gen_data_set(1500, absorptions_train,
                                   concentrations_train, backgrounds_train)

X_test, Y_test = gen_data_set(200, absorptions_test,
                                   concentrations_test, backgrounds_test)

#%% Saving the simulated data
dump(X_train, './training_data/X_train.joblib')
dump(Y_train, './training_data/Y_train.joblib')
dump(X_test, './training_data/X_test.joblib')
dump(X_test, './training_data/X_test .joblib')

#%% How does the original model work with noise?
Model_experiment_1 = load('./trained_models/experiment_1.joblib')
predictions_model_1 = Model_experiment_1.predict(X_test)
concentrations_plot(true_concentrations=Y_test, predictions=predictions_model_1,
                    labels=labels)


#%% Now we add baseline correction
Model_experiment_2_all_compounds = Model(model=PLSRegression(n_components=100, scale=False),
                                         target_normalization=True,
                                         baseline_correction=partial(SG_filter,
                                         window_length=75, poly_order=2, deriv_order=2))

Model_experiment_2_all_compounds.fit(X_train, Y_train)
predicted_concentrations = Model_experiment_2_all_compounds.predict(X_test)
concentrations_plot(true_concentrations=Y_test,
                    predictions=predicted_concentrations,
                    labels=labels)

# Save the model and the predictions
dump(Model_experiment_2_all_compounds, './trained_models/experiment_2_all_compounds.joblib')
dump(predicted_concentrations, './results/experiment_2_predicted_concentrations')