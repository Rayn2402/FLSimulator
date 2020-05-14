"""
    File containing all code related to machine learning models available for federated learning simulation

    NOTE :
            - L indicates the original dimension of feature vectors include in database X
            - M indicates the dimension of new feature vectors returned by phi
            - N indicates the number of feature vectors in the database X

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

learning_rate_choices = ['constant', 'adaptive', 'invscaling']


class LinearModel:

    def __init__(self, phi, M, eta0=1, learning_rate='invscaling'):

        """
        Linear Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the feature vector returned by phi (including bias)
        :param eta0: learning rate initial value in the training
        :param learning_rate: learning rate schedule

        """
        if eta0 <= 0:
            raise Exception("eta must be greater than 0")

        if learning_rate not in learning_rate_choices:
            raise Exception("learning rate choice not recognized")

        self.phi = phi
        self.eta = eta0
        self.w = self.__init_weight(M)
        self.lr_update = self.__init_learning_rate_function(eta0, learning_rate)

    @staticmethod
    def __init_weight(M):

        """
        Init weight vector with random values

        :param M: dimensions that w vector must satisfied
        :return: M X 1 numpy array
        """
        w = np.random.randn(M)
        w.resize((M, 1))
        return w

    def __init_learning_rate_function(self, eta0, lr_choice):

        if lr_choice == 'constant':

            def lr_update(last_loss, nb_w_update, X, t):
                pass

        elif lr_choice == 'invscaling':

            def lr_update(last_loss, nb_w_update, X, t):
                self.eta = eta0/(nb_w_update**0.25)

        else:

            def lr_update(last_loss, nb_w_update, X, t):
                if last_loss < self.loss(X, t):
                   self.eta /= 2

        return lr_update

    def train(self, X, t, nb_epoch, weight_init=None):

        """
        Train our linear model

        :param X: N x L numpy array with training data (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :param weight_init: Starting point of SGD in weight space (default = self.w)
        :return: updated weight vector (M x 1 numpy array)
        """

        # Warm start
        if weight_init is not None:
            self.w = weight_init

        # Variable ignition
        eta0 = self.eta
        last_loss = self.loss(X, t)
        nb_update = 0

        # Weights optimization
        for i in range(nb_epoch):
            X, t = shuffle(X, t)
            for n in range(X.shape[0]):
                self.update_w(X[n:n+1], t[n:n+1])
                self.lr_update(last_loss=last_loss, nb_w_update=(n+1+i*X.shape[0]), X=X, t=t)

        self.eta = eta0
        return self.w

    def loss(self, X, t):

        """
        Compute the loss associated with our current model

         :param X: N x L numpy array with training data (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :return: float
        """

        raise NotImplementedError

    def predict(self, x):

        """
        Predict the label for an input x

        :param x: 1 x L numpy array
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
        plt.plot(X, t, 'ro')
        plt.show()
        plt.close()


class SGDRegressor(LinearModel):

    def __init__(self, phi, M, eta0=1, learning_rate='invscaling'):

        """
        SGD Linear Regression Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the feature vector returned by phi (including bias)
        :param eta0: learning rate initial value in the training
        :param learning_rate: learning rate schedule
        """
        super().__init__(phi, M, eta0, learning_rate=learning_rate)

    def update_w(self, x_n, t_n):

        """
        Weight update in SGD

        :param x_n: n-th feature vector of our training data set (1 X L numpy array)
        :param t_n: label associated to the n-th feature vector (1 x 1 numpy array)
        :return: updated weight vector
        """
        self.w = self.w + self.eta*(t_n[0][0] - self.predict(x_n))*self.phi(x_n).transpose()

    def predict(self, x):

        """
        Predict the label for an input x

        :param x: 1 x M numpy array
        :return: predicted label
        """

        return np.dot(self.phi(x), self.w)[0][0]

    def loss(self, X, t, return_predictions=False):

        """
        Compute the least square loss associated with our current model

        :param X: N x L numpy array with training data (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :param return_predictions: bool that indicates if we returned the numpy array of predictions
        :return: float
        """
        pred = np.ndarray(shape=(X.shape[0], 1), buffer=np.array([self.predict(X[n:n+1]) for n in range(X.shape[0])]))
        loss = pred - t

        if return_predictions:
            return np.dot(loss.transpose(), loss), pred

        else:
            return np.dot(loss.transpose(), loss)


class SGDLogisticRegressor(LinearModel):

    def __init__(self, phi, M, eta0=1, learning_rate='invscaling'):
        """
        SGD Linear Regression Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the feature vector returned by phi (including bias)
        :param eta0: learning rate initial value in the training
        :param learning_rate: learning rate schedule
        """

        super().__init__(phi, M, eta0, learning_rate=learning_rate)

    def update_w(self, x_n, t_n):

        """
        Weight update in SGD

        :param x_n: n-th feature vector of our training data set (1 X L numpy array)
        :param t_n: label associated to the n-th feature vector (1 x 1 numpy array)
        :return: updated weight vector
        """
        self.w = self.w + self.eta*(self.predict(x_n) - t_n[0][0])*self.phi(x_n).transpose()

    def predict(self, x):

        """
        Predict the probability for an input x to have label 1

        :param x: 1 x M numpy array
        :return: predicted probability
        """
        return 1 / (1 + np.exp(np.dot(self.phi(x), self.w)[0][0]))

    def loss(self, X, t, return_predictions=False):

        """
        Computes the cross-entropy loss associated with our current model

        :param X: N x L numpy array with training data (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :param return_predictions: bool that indicates if we returned the numpy array of predictions
        :return: float
        """

        errors = []

        for n in range(X.shape[0]):

            t_n = t[n:n+1][0][0]
            y_n = self.predict(X[n:n+1])
            error = t_n*np.log(y_n) + (1-t_n)*np.log(1-y_n)
            errors.append(error)

        errors = np.array(errors)

        return -np.sum(errors)







