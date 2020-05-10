"""

    File managing all activities linked to data creation for simulation


"""
import random
import numpy as np
import matplotlib.pyplot as plt
from BasisFunctions import polynomial_features
from scipy.stats import beta

label_function_choices = ['linear', 'sin', 'tanh']


class OneDimensionalRDG:

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

        self.n = N
        self.noise = noise
        self.alpha = a
        self.beta = b
        self.labels = None
        self.label_function = self.generate_label_function(label_function)

    def generate_data(self, function=False):

        """
        Generate noisy labels associated with the features
        :return: N x 1 numpy array with feature vector, N x 1 numpy array with labels
        """
        # Generate features
        features = np.array(beta.rvs(a=self.alpha, b=self.beta, size=self.n))
        features.resize((self.n, 1))

        # Generate labels and add noise
        labels = self.label_function(features)
        random_noise = np.random.normal(loc=0, scale=self.noise, size=features.shape[0])
        random_noise.resize((features.shape[0], 1))
        labels += random_noise

        if function:
            return features, labels, self.label_function
        else:
            return features, labels

    @staticmethod
    def generate_label_function(choice):

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

    @staticmethod
    def plot_feature_distribution(X):

        """
        Shows an histogram of the feature distribution X

        :param X: N x 1 numpy array
        """

        plt.hist(X)
        plt.show()
        plt.close()

    @staticmethod
    def plot_labels(X, t, ground_truth_function=None):

        if ground_truth_function is not None:
            X_sample = np.linspace(0, 1, 500)
            X_sample.resize((500, 1))
            t_sample = ground_truth_function(X_sample)
            plt.plot(X_sample, t_sample)

        plt.plot(X, t, 'ro')
        plt.show()
        plt.close()









