"""
File containing multiple basis functions that can be used by our models
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def identity_with_bias(x):

    """
    Add a one at the beginning of the array

    :param x: 1 x N numpy array
    :return: 1 x (N + 1) numpy array
    """
    x = np.append(np.ones((x.shape[0], 1), dtype=int), x, axis=1)
    return x


def polynomial_features(deg):

    poly = PolynomialFeatures(degree=deg, include_bias=True)

    def polynomial(x):

        """
        Return polynomial features from original features x

        :param x: 1 x N numpy array
        :return: 1 x (nb new features) numpy array
        """
        return poly.fit_transform(x)

    return polynomial


