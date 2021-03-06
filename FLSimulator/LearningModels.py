"""
    File containing all code related to machine learning models available for federated learning simulation

    NOTE :
            - L indicates the original dimension of feature vectors include in X
            - M indicates the dimension of new feature vectors returned by phi
            - N indicates the number of feature vectors in X

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

        :param phi: basis function applied to training X
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
        Initializes weight vector with random values

        :param M: dimensions that w vector must satisfied
        :return: M X 1 numpy array
        """
        w = np.random.randn(M)
        w.resize((M, 1))
        self.w = w

    def __init_learning_rate_function(self, eta0, lr_choice):

        if lr_choice == 'constant':

            def lr_update(last_loss, epoch, X, t):
                return last_loss

        elif lr_choice == 'invscaling':

            def lr_update(last_loss, epoch, X, t):
                self.eta = eta0/(epoch**0.25)
                return last_loss

        else:
            def lr_update(last_loss, epoch, X, t):
                current_loss = self.loss(X, t)
                if last_loss < current_loss:
                    self.eta /= 2

                return current_loss

        return lr_update

    def train(self, X, t, nb_epoch, minibatch_size=1, weight_init=None):

        """
        Trains our linear model

        :param X: N x L numpy array with training X (L --> original number of feature)
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

            # We compute feature size applying phi function on first vector in Xset X
            feature_size = self.phi(X[0:1]).shape[1]
            self.init_weight(feature_size)

        # Variable ignition
        nb_minibatch = int(np.ceil(X.shape[0]/minibatch_size))
        eta0 = self.eta
        last_loss = float('inf')

        # X Shuffling
        X, t = shuffle(X, t)

        # Weights optimization
        for i in range(nb_epoch):

            for n in range(nb_minibatch):
                start = n*minibatch_size
                end = min(X.shape[0], (n+1)*minibatch_size)
                self.update_w(X[start:end], t[start:end])

            last_loss = self.lr_update(last_loss=last_loss, epoch=i+1, X=X, t=t)

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

         :param X: N x L numpy array with training X (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :return: float
        """

        raise NotImplementedError

    def predict(self, x):

        """
        Predicts the label for an input x

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

    def plot_model(self, X, t, title=None, save=False, save_path='', filename='model', save_format='.pdf'):

        """
        Plots the curve prediction of our model

        :param X: N x 1 numpy array with training X
        :param t: 1 x N numpy array with training labels
        :param title: title of the figure
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        :return:
        """
        raise NotImplementedError


class GDRegressor(LinearModel):

    def __init__(self, phi, eta0=1, learning_rate='invscaling'):

        """
        Linear Regression Machine Learning Models that trains with Gradient Descent

        :param phi: basis function applied to training X
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
        Predicts the label for an input x

        :param x: 1 x M numpy array
        :return: predicted label
        """

        return np.dot(self.phi(x), self.w)[0][0]

    def loss(self, X, t, return_predictions=False):

        """
        Computes the least square loss associated with our current model

        :param X: N x L numpy array with training X (L --> original number of feature)
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

    def plot_model(self, X, t, title=None, save=False, save_path='', filename='model', save_format='.pdf'):

        """
        Plots the curve prediction of our model (only available with 1-D feature space)

        :param X: N x 1 numpy array with training X
        :param t: 1 x N numpy array with training labels
        :param title: title of the figure
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """
        if X.shape[1] > 1:
            raise Exception('This function only accept feature spaces that are 1-D')

        x_sample = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)
        t_sample = [self.predict(np.array([[x]])) for x in x_sample]
        plt.plot(X, t, 'o')
        plt.plot(x_sample, t_sample, 'k')

        if title is not None:
            plt.title(title)

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

        plt.show()
        plt.close()


class LogisticRegressor(LinearModel):

    def __init__(self, phi, eta0=1, learning_rate='invscaling'):
        """
        SGD Logistic Regression Machine Learning Models for binary classification

        :param phi: basis function applied to training X
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
            weight_sum += (t_k[n:n + 1][0][0] - self.predict(X_k[n:n + 1])) * self.phi(X_k[n:n + 1]).transpose()

        self.w = self.w + self.eta * (1 / minibatch_size) * weight_sum

    def predict(self, x):

        """
        Predicts the probability for an input x to have label 1

        :param x: 1 x M numpy array
        :return: predicted probability
        """

        return 1 / (1 + np.exp(-1*np.dot(self.phi(x), self.w)[0][0]))

    def loss(self, X, t, return_predictions=False):

        """
        Computes the cross-entropy loss associated with our current model

        :param X: N x L numpy array with training X (L --> original number of feature)
        :param t: N x 1 numpy array with training labels
        :param return_predictions: bool that indicates if we returned the numpy array of predictions
        :return: float
        """

        errors = []
        for n in range(X.shape[0]):

            t_n = t[n:n+1][0][0]
            y_n = self.predict(X[n:n+1])

            # error = t_n*np.log(y_n) + (1-t_n)*np.log(1-y_n)
            if t_n == 1:
                error = np.log(y_n)

            elif t_n == 0:
                error = np.log(1-y_n)

            else:
                raise Exception('Label Error')

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

    def plot_model(self, X, t, title=None, save=False, save_path='', filename='model', save_format='.pdf'):

        """
        Plots the curve prediction of our model (only available with 1-D or 2-D feature space)
        The x-axis labels will be associated to w^t * phi(x)

        :param X: N x 1 or N x 2 numpy array with training X
        :param t: N x 1 numpy array with training labels
        :param title: title of the figure
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """
        if X.shape[1] > 2:
            raise Exception('This function only accept feature spaces that are 1-D or 2-D')

        # We make a copy of data with label added as a column
        a = np.hstack((X, t))
        a = a[a[:, X.shape[1]].argsort()]

        # We find index of class separation
        i = 0
        while a[i][X.shape[1]] == 0:
            i += 1

        if X.shape[1] == 1:

            # Points sampling for curve drawing
            x_sample = np.arange(a[:, 0].min(), a[:, 0].max(), 0.01)
            t_sample = [self.predict(np.array([[x]])) for x in x_sample]

            # Predictions of actual X dataset
            predictions = [self.predict(a[n:n + 1, [0]]) for n in range(X.shape[0])]

            # x axis labels computation for original data (w_transpose * phi(x))
            for n in range(a.shape[0]):
                a[n][0] = np.dot(self.phi(a[n:n+1, [0]]), self.w)[0][0]

            # Ground-truth points and their predictions
            plt.scatter(a[:i, 0], predictions[:i], edgecolors='k', label='0', alpha=0.5)
            plt.scatter(a[i:, 0], predictions[i:], edgecolors='k', label='1', alpha=0.5)

            # Axes config
            plt.legend(loc='upper left')
            plt.rc('text', usetex=True)
            plt.xlabel(r'$w^t \phi(x)$')
            plt.ylabel(r'$\sigma(w^t \phi(x))$')

            if title is not None:
                plt.title(title)

        else:

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3.5))

            # 1 - Class Separation

            # Points sampling for class contour drawing
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            x = np.arange(x_min, x_max, 0.02)
            y = np.arange(y_min, y_max, 0.02)

            # Generations of all possible pairs among x and y
            xx, yy = np.meshgrid(x, y)
            x_vis = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))])

            # Contour class coloring
            contour_out_pred = np.array([self.predict(x_vis[n:n+1]) for n in range(x_vis.shape[0])])
            contour_out = np.round(contour_out_pred)
            contour_out = contour_out.reshape(xx.shape)

            # We draw class separation in second subplot
            axes[1].contourf(xx, yy, contour_out, colors=('C0', 'k', 'C1', 'k'))
            axes[1].scatter(a[:i, 0], a[:i, 1], edgecolors='k')
            axes[1].scatter(a[i:, 0], a[i:, 1], edgecolors='k')

            # Axe configuration
            axes[1].set_xlim(x_min, x_max)
            axes[1].set_ylim(y_min, y_max)
            axes[1].set_xlabel('$X_1$')
            axes[1].set_ylabel('$X_2$')
            axes[1].set_title('Decision Boundary')

            # 2 - Predictions of actual X dataset
            for n in range(a.shape[0]):

                # We compute labels (w_transpose * phi(x)) and prediction associated to it
                new_x = np.dot(self.phi(a[n:n + 1, [0, 1]]), self.w)[0][0]
                a[n][1] = self.predict(a[n:n+1, [0, 1]])
                a[n][0] = new_x

            a = a[:, [0, 1]]

            axes[0].scatter(a[:i, 0], a[:i, 1], edgecolors='k', label='0', alpha=0.5)
            axes[0].scatter(a[i:, 0], a[i:, 1], edgecolors='k', label='1', alpha=0.5)

            # Axe configuration
            plt.rc('text', usetex=True)
            axes[0].set_title('Predictions')
            axes[0].legend(loc='upper left')
            axes[0].set_xlabel(r'$w^t \phi(x)$')
            axes[0].set_ylabel(r'$\sigma(w^t \phi(x))$')

            if title is not None:
                fig.suptitle(title)

            fig.tight_layout(h_pad=5, pad=3)

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

        plt.show()
        plt.close()



