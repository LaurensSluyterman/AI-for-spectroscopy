import numpy as np
from sklearn.model_selection import LeaveOneOut
from copy import deepcopy


class Model:
    """
    This class represents a trained model.

    Attributes:
        model: The actual model
        target_normalization (bool): Determines it the targets were normalized
            prior to training.
        baseline_correction: The type of baseline correction that is carried
            out prior to training.
        training_params: Other model-specific training parameters. This is
            needed to accommodate multiple different types of models.
    """
    def __init__(self, model, target_normalization=False,
                 baseline_correction=None, training_params=None):
        """
        Initialize the model

        Arguments:
            model: A model that at least has a fit method and a predict method.
                All sklearn models should work.
            target_normalization (bool): Determines it the targets get normalized
                prior to training. The default is not.
            baseline_correction: The baseline correction function. This function
                must take in a spectrum as input and output the baseline (NOT
                the corrected spectrum, this class does that step.
            training_params (dict): Additional model-specific training
                parameters. If not specified, it will be set to an empy
                dictionary.

        Raises:
            AttributeError: If the model does not contain a fit and predict
                method, an error will be raised.
        """
        self.model = model
        self.target_normalization = target_normalization
        self.baseline_correction = baseline_correction
        if training_params is None:
            self.training_params = {}
        else:
            self.training_params = training_params
        try:
            model.fit
        except AttributeError:
            print('The model has not fit method')
        try:
            model.predict
        except AttributeError:
            print('The model has not fit method')

    def fit(self, X_train, Y_train):
        """
        Train the model.

        if the model was initialized with target_normalization and/or
        baseline correction. This gets applied prior to training.

        Arguments:
            X_train (np.array): Array containing the covariates. Each
                row correspond to a data point.
            Y_train (np.array): Array containing the targets. Each row
                corresponds to a data point.
        """
        params = self.training_params
        X_train_new = deepcopy(X_train)
        Y_train_new = deepcopy(Y_train)
        if self.baseline_correction:
            for i, x in enumerate(X_train_new):
                X_train_new[i] = x - self.baseline_correction(x)
        if self.target_normalization:
            self.Y_mean_ = np.mean(Y_train, axis=0)
            self.Y_std_ = np.std(Y_train, axis=0)
            Y_train_new = (Y_train - self.Y_mean_) / self.Y_std_
        self.model.fit(X_train_new, Y_train_new, **params)

    def predict(self, X_test):
        """
        Make a prediction.

        If the class was initialized with baseline correction, this is applied
        prior to predicting. If the model was initialized with target
        normalization, the predictions get rescaled to the original scale.

        Arguments:
            X_test (np.array) : The covariates for which a prediction is made.

        Returns:
            predictions (np.array): The predictions on the original scale.
        """
        X_test_new = deepcopy(X_test)
        if self.baseline_correction:
            for i, x in enumerate(X_test_new):
                X_test_new[i] = x - self.baseline_correction(x)
        if self.target_normalization:
            predictions = self.model.predict(X_test_new) * self.Y_std_ + self.Y_mean_
            return predictions
        else:
            predictions = self.model.predict(X_test)
            return predictions


def leave_one_out_error(model_class, X_train, Y_train):
    """
    Perform leave_one_out CV.

    Parameters:
        model_class: An instance of our custom model class. Would also work
            with any other model that has a fit and predict instance.
        X_train (np.array): Array containing the covariates. Each
                row correspond to a data point.
        Y_train (np.array): Array containing the targets. Each row
                corresponds to a data point.

    Returns:
        mse_average: The average (taken over the folds) mean squared error.
            the mse is scaled to take the standard-deviation of the targets in
            to account such that every dimension of the output has the same
            order of importance.
    """
    mse_scores = []
    y_std = np.std(Y_train, axis=0)
    for train_index, test_index in LeaveOneOut().split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]

        model_class.fit(x_train, y_train)
        y_pred = model_class.predict(x_test)

        mse = np.mean(((y_test - y_pred) / y_std)**2)
        mse_scores.append(mse)
        mse_average = np.mean(mse_scores)
    return mse_average





