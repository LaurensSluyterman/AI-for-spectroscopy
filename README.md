This repository contains the code to create sythethic data sets by combining background measurements with simulated absorption spectra.

The relevant functions can be found in 'data_simulation.py'. The most relevant function within this file is the 'gen_dataset' function. This function takes as arguments a list of simulated
absorption coefficients, a second list with the corresponding concentration, and a third list with measured background spectra. These are then combined to create any desired
number of simulated absorbance spectra. Note that the absorption coefficient must be separately calculated, either using a database or software such as hapy. 

The file 'experiment_2.py' gives a demonstration of the entire proces including hyper-parameter optimization and a train/val/test split. 

The 'pipeline.py' file contains a Model-class that acts as a wrapper of sklearn models such that baseline correction and normalization can be implemented. A number of different
basic baseline corrections are implemented in 'baseline_correction.py' nad 'airPLS.py'.
