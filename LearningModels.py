"""
    File containing all code related to machine learning models available for federated learning simulation
"""
import random
import numpy as np


class LinearModel:

    def __init__(self, phi, phi_output_dim, eta):

        """
        Linear Machine Learning Models

        :param phi: basis function applied to training data
        :param phi_output_dim: dimension of the output vector returned by phi
        :param eta: learning rate during the training
        """
        if eta <= 0:
            raise Exception("eta must be greater than 0")

        self.phi = phi
        self.eta = eta
        self.w = self.__init_weight(phi_output_dim + 1)

    @staticmethod
    def __init_weight(weight_vec_dim):

        """
        Init weight vector with random values between -10 and 10

        :param weight_vec_dim: dimensions that w vector must satisfied
        :return: weight_vec_dim X 1 numpy array
        """
        w = np.array(random.sample(range(-10, 10), weight_vec_dim))
        w.resize((weight_vec_dim, 1))
        return w

    def __update_w(self, x_n, t_n):

        """
        Weight update in SGD

        :param x_n: n-th feature vector of our training data set
        :param t_n: label associated to the n-th feature vector
        :return: updated weight vector
        """
        raise NotImplementedError

    def train(self, X, t, new_weights=None):

        """
        Train our linear model

        :param X: N x M numpy array with training data
        :param t: N x 1 numpy array with training labels
        :param new_weights: Starting point of SGD in weights space (default = self.w)
        :return: updated weight vector
        """
        if new_weights is not None:
            self.w = new_weights

        for n in range(len(X)):
            self.__update_w(X[n], t[n])

        return self.w

    def plot_curve(self, X, t):

        """
        Plot the curve prediction of our model (only available with 1-D feature space)
        :param X: N x 1 numpy array with training data
        :param t: N x 1 numpy array with training labels
        :return:
        """
        raise NotImplementedError


class SGDRegressor(LinearModel):

    def __init__(self, phi, phi_output_dim, eta, l2_penalty=0):

        self.alpha = l2_penalty
        super().__init__(phi, phi_output_dim, eta)

    def __update_w(self, x_n, t_n):

        phi_n = self.phi(x_n)
        self.w = self.w + self.eta*(t_n - np.dot(self.w.transpose(), phi_n))*phi_n
