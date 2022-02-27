import numpy as np
from math import pi
import logging

class _NaiveBayesClassifier():

    def __init__(self):
        self.prior = None
        self.class_probability = None
        self.feature_mean = None
        self.feature_var = None
        self.unique_y = None
        self.id_2_class = None

    # Calculate the Gaussian probability distribution function for x
    @staticmethod
    def calculate_probability(x, mean, var):
        try:
            exponent = np.exp(-((x-mean)**2) / (2 * var))
        except ZeroDivisionError:
            return 0
        return 1 / np.sqrt(2 * pi * var) * exponent

    def cal_mean_variance(self, X_i, y_idx):
        """
        calculate mean and variance for each column features
        np.mean and np.var on axis = 0 -> calculate mean and variance for each column feauture on each observation
        """

        class_mean = np.mean(X_i, axis=0)
        class_var = np.var(X_i, axis=0)

        self.feature_mean[y_idx] = class_mean
        self.feature_var[y_idx] = class_var

    def fit(self, X, y):
        """
        It is used to calculate prior probability, feature_mean and feature_var
        """

        # set variables
        d_id_class = {}
        n_features = X.shape[1]
        n_classes = len(y)
        self.unique_y = np.unique(y)
        y_classes = len(self.unique_y)

        for index, y_ in enumerate(self.unique_y):
            d_id_class[index] = y_
        self.id_2_class = d_id_class

        # initialize prior, class probability
        self.prior = np.zeros(len(self.unique_y), dtype=np.float64)
        self.feature_mean = np.zeros((y_classes, n_features), dtype=np.float64)
        self.feature_var = np.zeros((y_classes, n_features), dtype=np.float64)

        # for each unique class, find out mean and variance
        for index, y_idx in enumerate(self.unique_y):
            logging.info(f"index: {index}, y_idx: {y_idx}")
            X_i = X[y == y_idx, :]
            logging.info(f"number of X_i: {X_i.shape[0]}")
            self.cal_mean_variance(X_i, y_idx)
            self.prior[index] = X_i.shape[0] / n_classes

        logging.info(f"feature_mean: {self.feature_mean}")
        logging.info(f"feature_var: {self.feature_var}")
        logging.info(f"prior: {self.prior}")

    def predict(self, X):

        unique_class = len(self.unique_y)
        arr_prop = np.zeros((unique_class, X.shape[0]))

        for idx in range(unique_class):
            logging.info(f"prior: {self.prior[idx]}")
            probabilities = self.prior[idx]
            mu = self.feature_mean[idx]
            sigma = self.feature_mean[idx]
            prop = self.calculate_probability(X, mu, sigma)
            probabilities *= np.prod(prop, axis=1) # each X record probability of a particular class
            arr_prop[idx] = probabilities

        # find out max probability across y classes
        max_prop = np.argmax(arr_prop, axis=0)
        logging.info(f"shape of max_prop: {max_prop.shape}")

        # return the predict label
        pred = np.ndarray(X.shape[0])
        for index in self.id_2_class:
            pred[max_prop == index] = self.id_2_class[index]

        return pred

    def predict_proba(self, X):

        unique_class = len(self.unique_y)
        arr_prop = np.zeros((unique_class, X.shape[0]))

        for idx in range(unique_class):
            logging.info(f"prior: {self.prior[idx]}")
            probabilities = self.prior[idx]
            mu = self.feature_mean[idx]
            sigma = self.feature_mean[idx]
            prop = self.calculate_probability(X, mu, sigma) # each X record probability of a particular class
            probabilities *= np.prod(prop, axis=1)
            arr_prop[idx] = probabilities

        # total probability value
        arr_sum_prop = arr_prop.sum(axis=0)
        exp_prop = arr_prop / arr_sum_prop

        return exp_prop.T