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
    def plot_feature_distribution(X, save=False, save_path='', filename='dist', save_format='.eps'):

        """
        Shows a density histogram of the feature distribution X

        :param X: N x 1 numpy array
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """

        plt.hist(X, color='C7', density=True)

        if save:
            plt.savefig(save_path+filename+save_format, format=save_format[1:])

        plt.show()
        plt.close()

    def plot_labels(self, X, t, add_ground_truth=False, save=False, save_path='',
                    filename='labels', save_format='.eps'):

        """
        Plots the data points (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """

        raise NotImplementedError

    @staticmethod
    def distribution_and_labels(X, t, title=None, save=False, save_path='',
                                filename='dist_and_labels', save_format='.eps'):
        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
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
        :param a: alpha parameter of the beta distribution that generates the feature
        :param b: beta parameter of the beta distribution that generates the feature
        :param label_function: choice of function that generates label associated to each feature
        """

        if label_function not in label_function_choices:
            raise Exception('Label function chosen is not recognized')

        super().__init__(a, b, noise)

        self.label_function = self.generate_label_function(label_function)

    def generate_data(self, N):

        """
        Generates features and noisy labels associated to them.

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

    def plot_labels(self, X, t, add_ground_truth=False, save=False, save_path='',
                    filename='labels', save_format='.eps'):

        """
        Plots the points (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """

        if add_ground_truth:
            self.plot_ground_truth()

        plt.plot(X, t, 'o')

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

        plt.show()
        plt.close()

    @staticmethod
    def distribution_and_labels(X, t, title=None, axes=None, save=False, save_path='', filename='dist_and_labels',
                                save_format='.eps'):

        """
        Plots a figure with both feature distribution and data points

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        :param axes: pyplot axes
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """
        show = False

        if axes is None:
            show = True
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
            axes[0].set_title('Labels')
            axes[1].set_title('Feature Distribution')

        if title is not None:
            axes[0].set_title(title)

        axes[0].plot(X, t, 'o')
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('')
        axes[1].hist(X, color='C7', density=True)
        axes[1].set_xlim(0, 1)

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

        if show:
            fig.tight_layout()
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

    def plot_labels(self, X, t, add_ground_truth=False, axe=None, save=False, save_path='', filename='labels',
                    save_format='.eps'):

        """
        Plots the labels point (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param axe: pyplot axe
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
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

            if save:
                plt.savefig(save_path + filename + save_format, format=save_format[1:])

            plt.show()
            plt.close()

    def distribution_and_labels(self, X, t, title=None, save=False, save_path='', filename='dist_and_labels',
                                save_format='.eps'):

        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
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

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

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
    def plot_labels(X, t, title='Class labels', x1_label='$X_1$', x2_label='$X_2$', axe=None, legend=False,
                    ylim=(-2, 2.8), save=False, save_path='', filename='labels', save_format='.eps'):

        """
        Plots the labels

        :param X: N x 2 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        :param x1_label: label associated to x-axis
        :param x2_label: label associated to y-axis
        :param axe: pyplot axe
        :param legend: bool indicating if we need the legend or not
        :param ylim: tuple with limits of y-axis
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
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
            axe.set_ylim(-2, 2.8)
            axe.set_ylabel(x2_label)
            axe.scatter(a[0:i, 0], a[0:i, 1], edgecolors='k', label='0')
            axe.scatter(a[i:, 0], a[i:, 1], edgecolors='k', label='1')
            if legend:
                axe.legend(loc='upper left')
        else:
            plt.title(title)
            plt.xlabel(x1_label)
            plt.ylim(ylim[0], ylim[1])
            plt.ylabel(x2_label)
            plt.scatter(a[0:i, 0], a[0:i, 1], edgecolors='k', label='0')
            plt.scatter(a[i:, 0], a[i:, 1], edgecolors='k', label='1')

            if legend:
                plt.legend(loc='upper left')

            if save:
                plt.savefig(save_path + filename + save_format, format=save_format[1:])

            plt.show()
            plt.close()

    @staticmethod
    def plot_feature_distribution(X, t, x1_title='$X_1$ marginal distributions',
                                  x2_title='$X_2$ marginal distributions', axes=None, legend=False,
                                  save=False, save_path='', filename='dist', save_format='.pdf'):

        """
        Shows an histogram of the feature distribution X

        :param X: N x 2 numpy array
        :param t: N x 1 numpy array
        :param x1_title: title of x1 feature distribution
        :param x2_title: title of x2 feature distribution
        :param axes: list with 2 pyplot axe
        :param legend: bool indicating if we want to show legend or not
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
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

        if x1_title is not None:
            axes[0].set_title(x1_title)

        axes[1].hist(a[0:i, 1], alpha=0.5, label='0', density=True)
        axes[1].hist(a[i:, 1], alpha=0.5, label='1', density=True)

        if x2_title is not None:
            axes[1].set_title(x2_title)

        if legend:
            axes[0].legend(loc='upper right')
            axes[1].legend(loc='upper right')

        if show:
            fig.tight_layout(h_pad=5, pad=3)

            if save:
                plt.savefig(save_path + filename + save_format, format=save_format[1:])

            plt.show()
            plt.close()

    @staticmethod
    def distribution_and_labels(Xs, ts, title=None, sub_height=1.25, sub_width=8, count_max=200,
                                save=False, save_path='', filename='dist_and_labels', save_format='.pdf'):

        """
        Plots a figure with both feature distribution and labels

        :param Xs: list of N x 2 numpy array
        :param ts: list of N x 1 numpy array
        :param sub_height: height of every subplot in the figure
        :param sub_width: width of each row in the figure
        :param count_max: maximum number of observations in one class
        :param title: plot title
        :param save: bool indicating if we want to save picture or not
        :param save_path: path indicating where we save the file if it is saved
        :param filename: name of the file if it is saved
        :param save_format: saving format
        """
        # Enable LaTeX
        plt.rc('text', usetex=True)

        # We save the number of X datasets
        n = len(Xs)

        # Set subplots
        fig, axes = plt.subplots(nrows=n, ncols=4, sharex='col', figsize=(sub_width, sub_height*n))

        if title is not None:
            fig.suptitle(title, y=1.025)

        # Set first line with titles
        if n > 1:
            axe = axes[0]
        else:
            axe = axes

        # Class Labels
        TwoClusterGenerator.plot_labels(Xs[0], ts[0], x1_label='', axe=axe[0], legend=True)

        # Densities
        TwoClusterGenerator.plot_feature_distribution(Xs[0], ts[0], axes=[axe[2], axe[3]])

        # Bar plot
        axe[1].bar(x=[0, 1], height=[(ts[0] == 0).sum(), (ts[0] == 1).sum()], color=['C0', 'C1'], edgecolor='k')
        axe[1].set_xticks([0, 1])
        axe[1].set_ylim(0, count_max)
        axe[1].set_title("Class count")

        # Subplots construction
        for i in range(1, n):

            # Class Labels
            if i == n-1:
                TwoClusterGenerator.plot_labels(Xs[i], ts[i], title='', axe=axes[i][0])
            else:
                TwoClusterGenerator.plot_labels(Xs[i], ts[i], title='', x1_label='', axe=axes[i][0])

            # Distributions
            TwoClusterGenerator.plot_feature_distribution(Xs[i], ts[i], x1_title=None, x2_title=None,
                                                          axes=[axes[i][2], axes[i][3]])

            # Bar plot
            axes[i][1].bar(x=[0, 1], height=[(ts[i] == 0).sum(), (ts[i] == 1).sum()], color=['C0', 'C1'], edgecolor='k')
            axes[i][1].set_ylim(0, count_max)
            axes[i][1].set_xticks([0, 1])

        fig.tight_layout(h_pad=0.1, w_pad=0.01)

        if save:
            plt.savefig(save_path + filename + save_format, format=save_format[1:])

        plt.show()
        plt.close()






















