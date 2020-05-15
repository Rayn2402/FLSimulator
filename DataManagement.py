"""

    File managing all activities linked to data creation for simulation


"""
import random
import numpy as np
import matplotlib.pyplot as plt
from BasisFunctions import polynomial_features
from scipy.stats import beta

label_function_choices = ['linear', 'sin', 'tanh']


class OneDimensionalDG:

    def __init__(self, N, a=1, b=1, noise=0):

        """
        One Dimension Data Generator

        :param N: number of 1-D feature vector needed
        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param noise: noise added to the data labels (definition depend on the One dimensional DG child class)

        """
        if not 0 <= noise <= 1:
            raise Exception('Noise value must be included between 0 and 1')
        self.n = N
        self.alpha = a
        self.beta = b
        self.noise = noise
        self.label_function = None

    def generate_data(self):

        """
        Generates labels associated with the features

        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels

        """
        raise NotImplementedError

    @staticmethod
    def plot_feature_distribution(X):

        """
        Shows an histogram of the feature distribution X

        :param X: N x 1 numpy array
        """

        plt.hist(X)
        plt.show()
        plt.close()

    def plot_labels(self, X, t, add_ground_truth=False):

        """
        Plots the labels point (x_n, t_n)

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param add_ground_truth: bool indicating if we should plot function used to generate labels
        """

        if add_ground_truth:
            X_sample = np.linspace(0, 1, 500)
            X_sample.resize((500, 1))
            t_sample = self.label_function(X_sample)
            plt.plot(X_sample, t_sample)

        plt.plot(X, t, 'ro')
        plt.show()
        plt.close()

    def distribution_and_labels(self, X, t, title=None):

        """
        Plots a figure with both feature distribution and labels

        :param X: N x 1 numpy array
        :param t: N x 1 numpy array
        :param title: plot title
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

        if title is not None:
            fig.suptitle(title)

        axes[0].plot(X, t, 'ro')
        axes[0].set_title('Labels')
        axes[1].hist(X)
        axes[1].set_title('Feature Distribution')
        fig.tight_layout()
        plt.show()
        plt.close()


class OneDimensionalRDG(OneDimensionalDG):

    def __init__(self, N, noise=0, a=1, b=1, label_function='linear'):

        """
        One Dimensional Regression Data Generator

        :param N: number of 1-D feature vector needed
        :param noise: standard deviation of the gaussian noise applied to the labels
        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param label_function: Choice of function to generate data
        """

        if label_function not in label_function_choices:
            raise Exception('Label function chosen is not recognized')

        super().__init__(N, a, b, noise)

        self.label_function = self.generate_label_function(label_function)

    def generate_data(self):

        """
        Generate noisy labels associated with the features

        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels
        """
        # Generate features
        features = np.array(beta.rvs(a=self.alpha, b=self.beta, size=self.n))
        features.resize((self.n, 1))

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


class OneDimensionalLRDG(OneDimensionalDG):

    def __init__(self, N, a=1, b=1, noise=0.10, increasing_prob=True, steepness=1):

        """
        One Dimension Logistic Regression DataGenerator

        :param N: number of 1-D feature vector needed
        :param a: alpha parameter of the beta distribution that generate the features
        :param b: beta parameter of the beta distribution that generate the features
        :param noise: standard deviation of the gaussian noise applied to the probabilities used to generate labels
        :param increasing_prob: indicates if the sigmoid used to generate labels is increasing or decreasing
        :param steepness: describes how steep is the sigmoid 1/(1+exp(-steepness*X))

        """
        super().__init__(N, a, b, noise)
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

    def generate_data(self):

        """
        Generates labels associated with the features

        :return: N x 1 numpy array with feature vectors, N x 1 numpy array with labels
        """

        # Generate features
        features = np.array(beta.rvs(a=self.alpha, b=self.beta, size=self.n))
        features.resize((self.n, 1))

        # Generate probabilities
        probability = self.label_function(features)

        # Generate noise to add to the probabilities
        random_noise = np.random.normal(loc=0, scale=self.noise, size=features.shape[0])
        random_noise.resize((features.shape[0], 1))
        probability += random_noise
        probability[probability < 0] = 0
        probability[probability > 1] = 1

        # Generate labels with a bernouilli
        labels = np.array([np.random.binomial(n=1, p=probability[n:n+1]) for n in range(self.n)])
        labels.resize((self.n, 1))

        return features, labels














