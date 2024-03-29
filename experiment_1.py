import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from pipeline import Model
from utils import resize, absorption_to_absorbance
from plot_functions import concentrations_plot

small_grid = np.loadtxt('PLS_data/small_grid.txt')

# training data
data = pd.read_csv('PLS_data/PLS_X2.txt', header=None)
absorption_spectra = data.to_numpy()[:, 1:].transpose()
concentrations_and_labels = pd.read_csv('PLS_data/PLS_Y2.txt', header=None)
concentrations = concentrations_and_labels.to_numpy()[:, 1:].transpose().astype(float)
labels = concentrations_and_labels.to_numpy()[:, 0].transpose()
wave_numbers = data.to_numpy()[:, 0]
absorption_spectra_resized = resize(small_grid,
                                    original_wave_numbers=wave_numbers,
                                    original_covariates=absorption_spectra)
X = absorption_to_absorbance(absorption_spectra_resized)
Y = concentrations

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#%% Saving the training and test data
np.savetxt('./training_data_no_background/X_train.txt', X_train, delimiter=',')
np.savetxt('./training_data_no_background/Y_train.txt', Y_train, delimiter=',')
np.savetxt('./training_data_no_background/X_test.txt', X_test, delimiter=',')
np.savetxt('./training_data_no_background/Y_test.txt', Y_test, delimiter=',')
np.savetxt('./training_data_no_background/labels.txt', labels, fmt='%s', delimiter=',')

#%% Fitting a basic PLS model
Model1 = Model(model=PLSRegression(n_components=30, scale=False),
               target_normalization=True,
               baseline_correction=None)

Model1.fit(X_train, Y_train)

# Plot the predictions against the true concentrations
Y_pred = Model1.predict(X_test)
concentrations_plot(Y_test, Y_pred, labels)

# Save the model and predictions
dump(Model1, './trained_models/experiment_1.joblib')
np.savetxt('./results/experiment_1_predictions.txt', Y_test, delimiter=',')
# loaded_model = load('./trained_models/experiment_1.joblib')

# Visualize the data
plt.plot(small_grid, X_test[4])
plt.ylabel('Absorbance')
plt.xlabel('wave_number (cm-1)')
plt.show()