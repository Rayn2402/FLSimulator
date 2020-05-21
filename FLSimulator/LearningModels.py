"""
    File containing all code related to machine learning models available for federated learning simulation

    NOTE :
            - L indicates the original dimension of feature vectors include in database X
            - M indicates the dimension of new feature vectors returned by phi
            - N indicates the number of feature vectors in the database X

"""

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

learning_rate_choices = ['constant', 'adaptive', 'invscaling']


class LinearModel:

    def __init__(self, phi, eta0=1, learning_rate='invscaling'):

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
        self.w = None
        self.lr_schedule = learning_rate
        self.lr_update = self.__init_learning_rate_function(eta0, learning_rate)

    def init_weight(self, M):

        """
        Init weight vector with random values

        :param M: dimensions that w vector must satisfied
        :return: M X 1 numpy array
        """
        w = np.random.randn(M)
        w.resize((M, 1))
        self.w = w

    def __init_learning_rate_function(self, eta0, lr_choice):

        if lr_choice == 'constant':

            def lr_update(last_loss, nb_w_update, X, t):
                pass

        elif lr_choice == 'invscaling':

            def lr_update(last_loss, nb_w_update, X, t):
                self.eta = eta0/(nb_w_update**0.25)

        else:
            def lr_update(last_loss, nb_w_update, X, t):
                current_loss = self.loss(X, t)
                if last_loss < current_loss:
                    self.eta /= 2

                last_loss = current_loss

        return lr_update

    def train(self, X, t, nb_epoch, minibatch_size=1, weight_init=None):

        """
        Train our linear model

        :param X: N x L numpy array with training data (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :param minibatch_size: size of minibatches used is gradient descent
        :param weight_init: Starting point of SGD in weight space (default = self.w)
        :return: updated weight vector (M x 1 numpy array)
        """

        # Warm start
        if weight_init is not None:
            self.w = weight_init

        # Weight ignition if not done yet
        if self.w is None:

            # We compute feature size applying phi function on first vector in dataset X
            feature_size = self.phi(X[0:1]).shape[1]
            self.init_weight(feature_size)

        # Variable ignition
        nb_minibatch = int(np.ceil(X.shape[0]/minibatch_size))
        eta0 = self.eta
        last_loss = self.loss(X, t)

        # Data Shuffling
        X, t = shuffle(X, t)

        # Weights optimization
        for i in range(nb_epoch):

            for n in range(nb_minibatch):
                start = n*minibatch_size
                end = min(X.shape[0], (n+1)*minibatch_size)
                self.update_w(X[start:end], t[start:end])
                self.lr_update(last_loss=last_loss, nb_w_update=(n+1+i*X.shape[0]), X=X, t=t)

        self.eta = eta0
        return self.w

    def reset_weights(self):

        """
        Resets the weights of the model

        """
        self.init_weight(len(self.w))

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

    def update_w(self, X_k, t_k):

        """
        Weight update in SGD of the k-th minibatch

        :param X_k: minibatch_size x L numpy array of features
        :param t_k: minibatch_size x 1 numpy array of labels
        :return: updated weight vector
        """
        raise NotImplementedError

    def copy(self):

        """
        Creates a copy of the model

        """
        return NotImplementedError

    def plot_model(self, X, t, title=None):

        """
        Plot the curve prediction of our model (only available with 1-D feature space)
        :param X: N x 1 numpy array with training data
        :param t: 1 x N numpy array with training labels
        :param title: title of the figure
        :return:
        """
        x_sample = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)
        t_sample = [self.predict(np.array([[x]])) for x in x_sample]
        plt.plot(X, t, 'ro')
        plt.plot(x_sample, t_sample, 'k')
        if title is not None:
            plt.title(title)
        plt.show()
        plt.close()


class GDRegressor(LinearModel):

    def __init__(self, phi, eta0=1, learning_rate='invscaling'):

        """
        Linear Regression Machine Learning Models that trains with Gradient Descent

        :param phi: basis function applied to training data
        :param M: dimension of the feature vector returned by phi (including bias)
        :param eta0: learning rate initial value in the training
        :param learning_rate: learning rate schedule
        """
        super().__init__(phi, eta0, learning_rate=learning_rate)

    def update_w(self, X_k, t_k):

        """
        Weight update in SGD of the k-th minibatch

        Gradient computation reference : Bishop, Pattern Recognition and Machine Learning, p.144

        :param X_k: minibatch_size x L numpy array of features
        :param t_k: minibatch_size x 1 numpy array of labels
        :return: updated weight vector
        """
        weight_sum = np.zeros((len(self.w), 1))
        minibatch_size = X_k.shape[0]

        for n in range(minibatch_size):
            weight_sum += (t_k[n:n+1][0][0] - self.predict(X_k[n:n+1]))*self.phi(X_k[n:n+1]).transpose()

        self.w = self.w + self.eta*(1/minibatch_size)*weight_sum

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
            return np.dot(loss.transpose(), loss)[0][0], pred

        else:
            return np.dot(loss.transpose(), loss)[0][0]

    def copy(self):

        """
        Creates a copy of the model

        """
        copy = GDRegressor(self.phi, self.eta, self.lr_schedule)
        copy.w = deepcopy(self.w)

        return copy


class LogisticRegressor(LinearModel):

    def __init__(self, phi, eta0=1, learning_rate='invscaling'):
        """
        SGD Linear Regression Machine Learning Models

        :param phi: basis function applied to training data
        :param M: dimension of the feature vector returned by phi (including bias)
        :param eta0: learning rate initial value in the training
        :param learning_rate: learning rate schedule
        """

        super().__init__(phi, eta0, learning_rate=learning_rate)

    def update_w(self, X_k, t_k):

        """
        Weight update in SGD of the k-th minibatch

        Gradient computation reference : Bishop, Pattern Recognition and Machine Learning, p.206

        :param X_k: minibatch_size x L numpy array of features
        :param t_k: minibatch_size x 1 numpy array of labels
        :return: updated weight vector
        """
        weight_sum = np.zeros((len(self.w), 1))
        minibatch_size = X_k.shape[0]

        for n in range(minibatch_size):
            weight_sum += (self.predict(X_k[n:n + 1]) - t_k[n:n + 1][0][0]) * self.phi(X_k[n:n + 1]).transpose()

        self.w = self.w + self.eta * (1 / minibatch_size) * weight_sum

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

    def copy(self):

        """
        Creates a copy of the model

        """
        copy = LogisticRegressor(self.phi, self.eta, self.lr_schedule)
        copy.w = deepcopy(self.w)

        return copy

    def plot_model(self, X, t, title=None):

        """
        Plot the curve prediction of our model (only available with 1-D or 2-D feature space)

        :param X: N x 1 numpy array with training data
        :param t: 1 x N numpy array with training labels
        :param title: title of the figure
        """
        # Points sampling for curve drawing
        x_sample = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)
        t_sample = [self.predict(np.array([[x]])) for x in x_sample]

        # Predictions of actual dataset
        predictions = [self.predict(X[n:n+1]) for n in range(X.shape[0])]

        # Curve drawing
        plt.plot(x_sample, t_sample, 'k')

        # Ground-truth points
        plt.scatter(X, predictions, c=t, edgecolors='k', cmap='bwr')

        if title is not None:
            plt.title(title)

        plt.show()
        plt.close()

















