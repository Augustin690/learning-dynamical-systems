import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator


# TODO: Define all the functions (in addition to the ones below) required to complete tasks 1-3 in this python file.

# Evaluation of a single Radial Basis Function (RBF)
def radial_basis_function(x, x_c, eps):
    """
    Compute radial basis function values as defined in the worksheet in section 1.2 (equation (7))

    Args:
    - x: Data points
    - x_c: Centers of the basis functions
    - eps: Epsilon parameter(bandwidth)
    
    Returns:
    - Radial basis function values evaluated at data points x
    """
    # Hint: Use cdist from scipy.spatial.distanc
    phi_x = np.exp(-(cdist(x_c, x, 'sqeuclid') / eps**2))

    return phi_x


def diameter(X):
    return np.max(X) - np.min(X)


def approx_non_linear_function(X, Y, L, ratio):
    x_c = np.linspace(np.min(X), np.max(Y), L).reshape((L, 1))
    X = X.reshape((X.shape[0], 1))
    eps = ratio * diameter(X)
    phi_x = radial_basis_function(X, x_c, eps)
    # return transform(X, least_squares(phi_x, Y))
    return phi_x


def built_int_interpolator(X, Y, eps):
    #X_augment = np.column_stack([X, np.ones(X.shape)])

    return RBFInterpolator(X, Y, kernel='gaussian', epsilon= eps)(X)


def least_squares(A, b, cond=0.1):
    """ Returns a Least squares solution of Ax = b

    Args:
        A (npt.NDArray[np.float32]): Input array for the least squares problem
        b (npt.NDArray[np.float32]): Output array for the least squares problem
        cond : cut-off ratio for singular values of a
    Returns:
        npt.NDArray[np.float32]: Least squares solution of Ax = b
    """
    # TODO: Implement using scipy.linalg.lstsq 
    # Hint: Don't forget that lstsq also returns other parameters such as residuals, rank, singular values etc.
    # Solve the least squares problem using scipy.linalg.lstsq
    A_augment = np.column_stack([A, np.ones(A.shape)])
    x, residuals, rank, s = lstsq(A_augment, b, cond=cond)

    # Return the solution
    return x


def transform(X, coefficients):
    X_augment = np.column_stack([X, np.ones(X.shape)])
    #return coefficients @ X_augment.T
    print('updated')
    return X_augment@coefficients
