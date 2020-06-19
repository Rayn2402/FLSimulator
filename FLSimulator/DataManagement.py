"""

    File managing all activities linked to data creation for simulation


"""
import random
import numpy as np
import matplotlib.pyplot as plt
from .BasisFunctions import polynomial_features
from scipy.stats import beta, multivariate_normal

label_function_choices = ['linear', 'sin', 'tanh']


class OneDimensionalDG:

    def __init__(self, a=1, b=1, noise=0):

        """
        One Dimension Data Generator

        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param noise: noise added to the data labels (definition depend on the One dimensional DG child class)

        """
        if not 0 <= noise <= 1:
            raise Exception('Noise value must be included between 0 and 1')

        self.alpha = a
        self.beta = b
        self.noise = noise
        self.label_function = None

    def generate_data(self, N):

        """
        Generates labels associated with the features

        :param N: number of 1-D feature vector needed
        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels

        """
        raise NotImplementedError

    @staticmethod
    def plot_feature_distribution(X):

        """
        Shows an histogram of the feature distribution X

        :param X: N x 1 numpy array
        """

        plt.hist(X, color='C7', density=True)
        plt.show()
        plt.close()

    def plot_labels(self, X, t, add_ground_truth=False):

        """
        Plots the labels point (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels

        """

        raise NotImplementedError

    @staticmethod
    def distribution_and_labels(X, t, title=None):
        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        """

        raise NotImplementedError

    def plot_ground_truth(self):

        """
        Plots the ground truth curve used to generate labels

        """
        X_sample = np.linspace(0, 1, 500)
        X_sample.resize((500, 1))
        t_sample = self.label_function(X_sample)
        plt.plot(X_sample, t_sample, 'k')


class OneDimensionalRDG(OneDimensionalDG):

    def __init__(self, noise=0.10, a=1, b=1, label_function='linear'):

        """
        One Dimensional Regression Data Generator

        :param noise: standard deviation of the gaussian noise applied to the labels
        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param label_function: Choice of function to generate data
        """

        if label_function not in label_function_choices:
            raise Exception('Label function chosen is not recognized')

        super().__init__(a, b, noise)

        self.label_function = self.generate_label_function(label_function)

    def generate_data(self, N):

        """
        Generates noisy labels associated with the features

        :param N: number of 1-D feature vector needed
        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels
        """
        # Generate features
        features = np.array(beta.rvs(a=self.alpha, b=self.beta, size=N))
        features.resize((N, 1))

        # Generate labels and add noise
        labels = self.label_function(features)
        random_noise = np.random.normal(loc=0, scale=self.noise, size=features.shape[0])
        random_noise.resize((features.shape[0], 1))
        labels += random_noise

        return features, labels

    @staticmethod
    def generate_label_function(choice):

        """
        Generates the function that will produce labels associated with the features

        :param choice: label_function choice in ['linear', 'sin', 'tanh']
        :return: function
        """

        if choice == 'sin':

            # Return sin(X) function
            def f(X):
                return np.sin(X*2*np.pi)

        elif choice == 'tanh':

            # Return tanh function
            def f(X):
                return np.tanh(X*np.pi)

        else:

            # Return a random linear function
            p = polynomial_features(1)
            coefs = np.array([random.uniform(0, 5) for i in range(2)])
            coefs.resize(2, 1)

            def f(X):
                return np.matmul(p(X), coefs)

        return f

    def plot_labels(self, X, t, add_ground_truth=False):

        """
        Plots the labels point (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        """

        if add_ground_truth:
            self.plot_ground_truth()

        plt.plot(X, t, 'o')
        plt.show()
        plt.close()

    @staticmethod
    def distribution_and_labels(X, t, title=None):

        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

        if title is not None:
            fig.suptitle(title)

        axes[0].plot(X, t, 'o')
        axes[0].set_title('Labels')
        axes[0].set_xlim(0, 1)
        axes[1].hist(X, color='C7', density=True)
        axes[1].set_title('Feature Density')
        axes[1].set_xlim(0, 1)
        fig.tight_layout(h_pad=5, pad=3)
        plt.show()
        plt.close()


class OneDimensionalLRDG(OneDimensionalDG):

    def __init__(self, a=1, b=1, noise=0.10, increasing_prob=True, steepness=1):

        """
        One Dimension Logistic Regression DataGenerator

        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param noise: standard deviation of the gaussian noise applied to the probabilities used to generate labels
        :param increasing_prob: indicates if the sigmoid used to generate labels is increasing or decreasing
        :param steepness: describes how steep is the sigmoid 1/(1+exp(-steepness*X))

        """
        super().__init__(a, b, noise)
        self.label_function = self.generate_label_function(increasing_prob, steepness)

    @staticmethod
    def generate_label_function(increasing_prob, steepness):

        """
        Generates the function that will produce labels associated with the features

        :return: function

        """
        if increasing_prob:
            m = -steepness
        else:
            m = steepness

        def f(X):
            return 1/(1+np.exp(m*(X-0.5)*10))

        return f

    def generate_data(self, N):

        """
        Generates labels associated with the features

        :param N: number of 1-D feature vector needed
        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels
        """

        # Generate features
        features = np.array(beta.rvs(a=self.alpha, b=self.beta, size=N))
        features.resize((N, 1))

        # Generate probabilities
        probability = self.label_function(features)

        # Generate noise to add to the probabilities
        random_noise = np.random.normal(loc=0, scale=self.noise, size=features.shape[0])
        random_noise.resize((features.shape[0], 1))
        probability += random_noise
        probability[probability < 0] = 0
        probability[probability > 1] = 1

        # Generate labels with a bernouilli
        labels = np.array([np.random.binomial(n=1, p=probability[n:n+1]) for n in range(N)])
        labels.resize((N, 1))

        return features, labels

    def plot_labels(self, X, t, add_ground_truth=False, axe=None):

        """
        Plots the labels point (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        """

        if add_ground_truth:
            self.plot_ground_truth()

        # We make a copy of data with label added as a column
        a = np.hstack((X, t))
        a = a[a[:, 1].argsort()]

        # We find index of class separation
        i = 0
        while a[i][1] == 0:
            i += 1

        if axe is not None:
            axe.scatter(a[:i, 0], a[:i, 1], edgecolors='k')
            axe.scatter(a[i:, 0], a[i:, 1], edgecolors='k')
        else:
            plt.scatter(a[:i, 0], a[:i, 1], edgecolors='k')
            plt.scatter(a[i:, 0], a[i:, 1], edgecolors='k')
            plt.show()
            plt.close()

    def distribution_and_labels(self, X, t, title=None):

        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        """

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 3))

        if title is not None:
            fig.suptitle(title)

        # Labels
        self.plot_labels(X, t, axe=axes[0])
        axes[0].set_title('Labels')

        # Histogram
        axes[1].hist(X, color='C7', density=True)
        axes[1].set_title('Feature Distribution')

        # Bar plot
        axes[2].bar(x=[0, 1], height=[(t == 0).sum(), (t == 1).sum()], color=['C0', 'C1'], edgecolor='k')
        axes[2].set_xticks([0, 1])
        axes[2].set_title('Count')

        fig.tight_layout(pad=3, h_pad=4)
        plt.show()
        plt.close()


class TwoClusterGenerator:

    @staticmethod
    def generate_data(sample_sizes, centers, covs):

        """
        Generates the clusters

        :param sample_sizes: list with number of samples to produce for each class
        :param centers: list with two 1x2 numpy arrays representing mean of both class distributions
        :param covs: list with two 2x2 numpy array representing covariance matrix of both distribution

        :return:

        """
        if len(sample_sizes) != 2:
            raise Exception("size of argument does not match with the number of classes")

        X = multivariate_normal.rvs(mean=centers[0],
                                    cov=covs[0],
                                    size=sample_sizes[0])

        t = np.zeros((sample_sizes[0], 1))

        X = np.append(X, multivariate_normal.rvs(mean=centers[1],
                                                 cov=covs[1],
                                                 size=sample_sizes[1]), axis=0)

        t = np.append(t, np.ones((sample_sizes[1], 1)), axis=0)

        return X, t

    @staticmethod
    def plot_labels(X, t, title='Labels', x1_label='$X_1$', x2_label='$X_2$', axe=None, legend=False):

        """
        Plots the labels

        :param X: N x 2 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        :param x1_label: label associated to x-axis
        :param x2_label: label associated to y-axis
        :param axe: pyplot axe
        :param legend: bool indicating if we need the legend or not
        """
        # Enable LaTeX
        plt.rc('text', usetex=True)

        # We make a copy of data with label added as a column
        a = np.hstack((X, t))
        a = a[a[:, 2].argsort()]

        # We find index of class separation
        i = 0
        while a[i][2] == 0:
            i += 1

        if axe is not None:
            axe.set_title(title)
            axe.set_xlabel(x1_label)
            axe.set_ylabel(x2_label)
            axe.scatter(a[0:i, 0], a[0:i, 1], edgecolors='k', label='0')
            axe.scatter(a[i:, 0], a[i:, 1], edgecolors='k', label='1')
            if legend:
                axe.legend(loc='upper left')
        else:
            plt.title(title)
            plt.xlabel(x1_label)
            plt.ylabel(x2_label)
            plt.scatter(a[0:i, 0], a[0:i, 1], edgecolors='k', label='0')
            plt.scatter(a[i:, 0], a[i:, 1], edgecolors='k', label='1')
            if legend:
                plt.legend(loc='upper left')
            plt.show()
            plt.close()

    @staticmethod
    def plot_feature_distribution(X, t, x1_title='$X_1$ marginal density',
                                  x2_title='$X_2$ marginal density', axes=None):

        """
        Shows an histogram of the feature distribution X

        :param X: N x 2 numpy array
        :param t: N x 1 numpy array
        :param x1_title: title of x1 feature distribution
        :param x2_title: title of x2 feature distribution
        :param axes: list with 2 pyplot axe
        """
        # Enable LaTeX
        plt.rc('text', usetex=True)

        # We make a copy of data with label added as a column
        a = np.hstack((X, t))
        a = a[a[:, 2].argsort()]

        # We find index of class separation
        i = 0
        while a[i][2] == 0:
            i += 1

        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
            show = True

        else:
            show = False

        axes[0].hist(a[0:i, 0], alpha=0.5, label='0', density=True)
        axes[0].hist(a[i:, 0], alpha=0.5, label='1', density=True)
        axes[0].legend(loc='upper right')
        axes[0].set_title(x1_title)

        axes[1].hist(a[0:i, 1], alpha=0.5, label='0', density=True)
        axes[1].hist(a[i:, 1], alpha=0.5, label='1', density=True)
        axes[1].legend(loc='upper right')
        axes[1].set_title(x2_title)

        if show:
            fig.tight_layout(h_pad=5, pad=3)
            plt.show()
            plt.close()

    @staticmethod
    def distribution_and_labels(X, t, title=None):

        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        """
        # Enable LaTeX
        plt.rc('text', usetex=True)

        # Set subplots
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

        if title is not None:
            fig.suptitle(title)

        # Labels
        TwoClusterGenerator.plot_labels(X, t, axe=axes[0])

        # Histogram
        TwoClusterGenerator.plot_feature_distribution(X, t, axes=[axes[1], axes[2]])

        # Bar plot
        axes[3].bar(x=[0, 1], height=[(t == 0).sum(), (t == 1).sum()], color=['C0', 'C1'], edgecolor='k')
        axes[3].set_xticks([0, 1])
        axes[3].set_title('Labels Count')

        fig.tight_layout(h_pad=5, pad=3)
        plt.show()
        plt.close()



















