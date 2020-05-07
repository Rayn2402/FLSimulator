"""
    File containing all code related to machine learning models available for federated learning simulation
"""
import random
import numpy as np


class LinearModels:

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

        return np.array(random.sample(range(-10, 10), weight_vec_dim))

    def train(self, X, t):

        """

        :param X: N x M numpy array with training data
        :param t: 1 x N numpy array with training labels
        :return: updated weights vector
        """

        raise NotImplementedError
