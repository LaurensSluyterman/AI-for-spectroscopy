import numpy as np
import pandas as pd
import scipy
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import resize
from data_simulation import gen_data_set

# The desired grid for the simulated data set
target_grid = np.loadtxt('PLS_data/small_grid.txt')

# The simulated absorbance spectra from, for example, HITRAN=
data = pd.read_csv('SCFA_data/SCFA_Aspectra.txt', header=None)
wave_numbers = data.to_numpy()[:, 0]  # In our data, the first column of the spectra file contained the wave numbers
absorption_spectra = data.to_numpy()[:, 1:].transpose()  # The other columns contain the actual absorption coefficients
concentrations_and_labels = pd.read_csv('SCFA_data/SCFA_concentrations.txt', header=None)
concentrations = concentrations_and_labels.to_numpy()[:, 1:].transpose()
labels = concentrations_and_labels.to_numpy()[:, 0].transpose()

absorption_spectra_resized = resize(target_grid,
                                    original_wave_numbers=wave_numbers,
                                    original_covariates=absorption_spectra)  # Resize to desired grid

# Import background spectra and rescale to size of target grid
directory = './background_spectra_2'
background_names = file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
backgrounds = [scipy.io.loadmat(f'./{directory}/{name}')['spec_sig_full_reduced'].T for name in background_names]
wave_numbers_background = scipy.io.loadmat("background-spectra/wavenumber_grid.mat")['wavenumber_grid'][0]


# Resize the backgrounds to the target grid
for i, background in enumerate(backgrounds):
    backgrounds[i] = resize(target_grid,
                            original_wave_numbers=wave_numbers_background,
                            original_covariates=background)


# Make train/val/test split of the backgrounds and absorption/concentrations
backgrounds_trainval, backgrounds_test = train_test_split(backgrounds, test_size=7, random_state=2)
backgrounds_train, backgrounds_val = train_test_split(backgrounds_trainval, test_size=7, random_state=2)

absorptions_trainval, absorptions_test, concentrations_trainval, concentrations_test = train_test_split(absorption_spectra_resized,
                                                                                                        concentrations,
                                                                                                        random_state=2,
                                                                                                        test_size=150)
absorptions_train, absorptions_val, concentrations_train, concentrations_val = train_test_split(absorptions_trainval,
                                                                                                concentrations_trainval,
                                                                                                random_state=2,
                                                                                                test_size=150)
# Create the final training/val/test sets
np.random.seed(34)
X_train, Y_train = gen_data_set(1500, absorptions_train, concentrations_train,
                                backgrounds_train)

X_val, Y_val = gen_data_set(500, absorptions_val, concentrations_val,
                            backgrounds_val)

X_test, Y_test = gen_data_set(200, absorptions_test, concentrations_test,
                              backgrounds_test)

#%% Saving the simulated data
np.savetxt('./SCFA_data/X_train.txt', X_train, delimiter=',')
np.savetxt('./SCFA_data/Y_train.txt', Y_train, delimiter=',')
np.savetxt('./SCFA_data/X_val.txt', X_val, delimiter=',')
np.savetxt('./SCFA_data/Y_val.txt', Y_val, delimiter=',')
np.savetxt('./SCFA_data/X_test.txt', X_test, delimiter=',')
np.savetxt('./SCFA_data/Y_test.txt', Y_test, delimiter=',')
np.savetxt('./SCFA_data/labels.txt', labels, fmt='%s', delimiter=',')

#%% Visual check
plt.plot(target_grid, X_test[0])
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
plt.show()
