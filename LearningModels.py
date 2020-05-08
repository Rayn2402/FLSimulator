"""
    File containing all code related to machine learning models available for federated learning simulation
"""
import random
import numpy as np
from matplotlib import pyplot as plt


class LinearModel:

    def __init__(self, phi, M, eta):

        """
        Linear Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the output vector returned by phi (including bias)
        :param eta: learning rate during the training
        """
        if eta <= 0:
            raise Exception("eta must be greater than 0")

        self.phi = phi
        self.eta = eta
        self.w = self.__init_weight(M)

    @staticmethod
    def __init_weight(weight_vec_dim):

        """
        Init weight vector with random values between -10 and 10

        :param weight_vec_dim: dimensions that w vector must satisfied
        :return: weight_vec_dim X 1 numpy array
        """
        w = np.array(random.sample(range(1, 10), weight_vec_dim))
        w.resize((weight_vec_dim, 1))
        return w

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
            self.update_w(X[n], t[n])

        return self.w

    def predict(self, x):

        """
        Predict the label for an input x

        :param x: 1 x N numpy array
        :return: predicted label
        """

        raise NotImplementedError

    def update_w(self, x_n, t_n):

        """
        Weight update in SGD

        :param x_n: n-th feature vector of our training data set
        :param t_n: label associated to the n-th feature vector
        :return: updated weight vector
        """
        raise NotImplementedError

    def plot_curve(self, X, t, start, stop):

        """
        Plot the curve prediction of our model (only available with 1-D feature space)
        :param X: N x 1 numpy array with training data
        :param t: 1 x N numpy array with training labels
        :param start: starting point on x-axis
        :param stop: ending point on x-axis
        :return:
        """
        x_sample = np.arange(start, stop, 0.01)
        t_sample = [self.predict(np.array([[x]])) for x in x_sample]
        plt.plot(x_sample, t_sample)
        return plt.show()


class SGDRegressor(LinearModel):

    def __init__(self, phi, M, eta, lamb=0):

        """
        SGD Linear Regression Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the output vector returned by phi (including bias)
        :param eta: learning rate during the training
        :param lamb: parameter in regularization term added to least square lost "lamb*(w_transpose)*(w)"
        """

        self.lamb = lamb
        super().__init__(phi, M, eta)

    def update_w(self, x_n, t_n):

        """
        Weight update in SGD

        :param x_n: n-th feature vector of our training data set (1 X N numpy array)
        :param t_n: label associated to the n-th feature vector (1 x 1 numpy array)
        :return: updated weight vector
        """
        self.w = self.w + self.eta*(t_n[0] - self.predict(x_n))*self.phi(x_n).transpose() - self.lamb*self.w

    def predict(self, x):

        """
        Predict the label for an input x
        :param x: 1 x N numpy array
        :return: predicted label
        """
        return np.dot(self.phi(x), self.w)[0][0]
