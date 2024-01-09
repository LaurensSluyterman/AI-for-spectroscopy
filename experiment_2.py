import numpy as np
import pandas as pd
import scipy
import os
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
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


backgrounds_train, backgrounds_test = train_test_split(backgrounds, test_size=7, random_state=2)
backgrounds_train, backgrounds_val = train_test_split(backgrounds_train, test_size=7, random_state=2)

absorptions_train, absorptions_test, concentrations_train, concentrations_test = train_test_split(absorption_spectra_resized,
                                                                                                  concentrations,
                                                                                                  random_state=2,
                                                                                                  test_size=150)
absorptions_train, absorptions_val, concentrations_train, concentrations_val = train_test_split(absorptions_train,
                                                                                                  concentrations_train,
                                                                                                  random_state=2,
                                                                                                  test_size=150)
np.random.seed(34)
X_train, Y_train = gen_data_set(1500, absorptions_train,
                                   concentrations_train, backgrounds_train)

X_val, Y_val = gen_data_set(500, absorptions_val,
                                   concentrations_val, backgrounds_val)

X_test, Y_test = gen_data_set(200, absorptions_test,
                                   concentrations_test, backgrounds_test)

#%% Saving the simulated data
dump(X_train, './training_data/X_train.joblib')
dump(Y_train, './training_data/Y_train.joblib')
dump(X_val, './training_data/X_val.joblib')
dump(Y_val, './training_data/Y_val.joblib')
dump(X_test, './training_data/X_test.joblib')
dump(Y_test, './training_data/Y_test.joblib')
dump(labels, './training_data/labels.joblib')

np.savetxt('./training_data/X_train.txt', X_train, delimiter=',')
np.savetxt('./training_data/Y_train.txt', Y_train, delimiter=',')
np.savetxt('./training_data/X_test.txt', X_val, delimiter=',')
np.savetxt('./training_data/Y_test.txt', Y_val, delimiter=',')
np.savetxt('./training_data/X_test.txt', X_test, delimiter=',')
np.savetxt('./training_data/Y_test.txt', Y_test, delimiter=',')
np.savetxt('./training_data/labels.txt', labels, fmt='%s', delimiter=',')

# X_train = load('./training_data/X_train.joblib')
# Y_train = load('./training_data/Y_train.joblib')
# X_test = load('./training_data/X_test.joblib')
# Y_test = load('./training_data/Y_test.joblib')
# labels =load('./training_data/labels.joblib')
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

#%% Hyperparameter optimization
Y_val_normalized = (Y_val - np.mean(Y_train, axis=0)) / np.std(Y_train, axis=0)
best_score = 1e10
component_options = [10, 15, 25, 50, 75, 100]
length_options = [50, 75, 100, 125, 150, 175, 200]

for i, n in enumerate(component_options):
    print(f'{i+1} of {len(component_options)}')
    for length in length_options:
        test_model = Model(model=PLSRegression(n_components=n, scale=False),
                           target_normalization=True,
                           baseline_correction=partial(SG_filter, window_length=length,
                                                       poly_order=2, deriv_order=2))
        test_model.fit(X_train, Y_train)
        predictions = test_model.predict(X_val)
        predictions_normalized = (predictions - np.mean(Y_train, axis=0)) / np.std(Y_train, axis=0)
        assert np.shape(predictions_normalized) == np.shape(Y_val_normalized)
        score = np.sqrt(np.mean(np.square(predictions_normalized - Y_val_normalized)))
        if score < best_score:
            print(f'Components:{n}, Length:{length}, Score:{score}')
            best_model = test_model
            best_score = score
            best_n = n
            best_length = length

Model_experiment_2_all_compounds = Model(model=PLSRegression(n_components=best_n, scale=False),
                                         target_normalization=True,
                                         baseline_correction=partial(SG_filter,
                                         window_length=best_length, poly_order=2, deriv_order=2))

Model_experiment_2_all_compounds.fit(X_train, Y_train)
predicted_concentrations = Model_experiment_2_all_compounds.predict(X_test)
rmse_values = np.sqrt(np.mean(np.square(predicted_concentrations - Y_test), axis=0))
np.sqrt(np.mean(np.square((predicted_concentrations - Y_test) / np.std(Y_train, axis=0))))
for i, label in enumerate(labels):
    print(f'rmse {label}: {rmse_values[i]}')
concentrations_plot(true_concentrations=Y_test,
                    predictions=predicted_concentrations,
                    labels=labels)


#%% Hyperparamter optimization if all we want is to have acetone correct
best_score_acetone = 1e10

for i, n in enumerate(component_options):
    print(f'{i + 1} of {len(component_options)}')
    for length in length_options:
        test_model = Model(model=PLSRegression(n_components=n, scale=False),
                           target_normalization=True,
                           baseline_correction=partial(SG_filter, window_length=length,
                                                       poly_order=2, deriv_order=2))
        test_model.fit(X_train, Y_train)
        predictions = test_model.predict(X_val)
        predictions_normalized = (test_model.predict(X_val) - np.mean(Y_train, axis=0)) / np.std(Y_train, axis=0)
        assert np.shape(predictions_normalized[:, 8]) == np.shape(Y_val_normalized[:, 8])
        score = np.sqrt(np.mean(np.square(predictions_normalized[:, 8] - Y_val_normalized[:, 8])))
        if score < best_score_acetone:
            print('Found new best:')
            print(f'Components:{n}, Length:{length}, Score:{score}')
            best_model_acetone = test_model
            best_score_acetone = score
            best_n_acetone = n
            best_length_acetone = length

Model_experiment_2_acetone = Model(model=PLSRegression(n_components=best_n_acetone, scale=False),
                                   target_normalization=True,
                                   baseline_correction=partial(SG_filter,
                                                               window_length=best_length_acetone,
                                                               poly_order=2,
                                                               deriv_order=2))

Model_experiment_2_acetone.fit(X_train, Y_train)
predicted_concentrations = Model_experiment_2_acetone.predict(X_test)
concentrations_plot(true_concentrations=Y_test,
                    predictions=predicted_concentrations,
                    labels=labels)

np.sqrt(np.mean(np.square(predicted_concentrations[:, 8] - Y_test[:, 8])))
