import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from scipy.integrate import solve_ivp


# TODO: Define all the functions (in addition to the ones below) required to complete tasks 1-3 in this python file.


# Evaluation of a single Radial Basis Function (RBF)
def radial_basis_function(x, x_c, eps):
    """
    Compute radial basis function values as defined in the worksheet in section 1.2 (equation (7))

    Args:
        x: Data points
        x_c: Centers of the basis functions
        eps: Epsilon parameter(bandwidth)

    Returns:
        Radial basis function values evaluated at data points x
    """
    # Hint: Use cdist from scipy.spatial.distanc
    phi_x = np.exp(-(cdist(x_c, x, "sqeuclid") / eps**2))

    return phi_x.T


def least_squares(A, b, cond=0.01):
    """Returns a Least squares solution of Ax = b

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

    # Return the solution
    return lstsq(A, b, cond=cond)


def linear_fit(X, Y):
    """Fits linear model between X (input data) and Y (dependent data)

    Args:
        X: Input array
        Y: Dependant array, f(X)
    Returns:
        npt.NDArray[np.float64]: Least squares coefficients
    """
    return least_squares(X, Y)[0]


def transform(X, C):
    """Applies changes to input data X according to the coefficient of the newly fit linear model

    Args:
        X: Input array to transform
        C: Coefficient array solution to least squares problem CX = Y
    Returns:
        npdt.NDArray[np.float64]: Transformation of X according to linear model == approximation of Y
    """
    return np.dot(X, C)


def linear_fit_transform(X, Y):
    """Fits linear model on X,Y then apply transformation on X using coefficients of newly fit model

    Args:
        X: Input array
        Y: Dependent array, f(X)
    Returns:
        npt.NDArray[np.float64]: transformation of X according to linear model
    """
    coef = linear_fit(X, Y)
    return transform(X, coef), coef


def non_linear_fit_transform(X, Y, x_c, ratio):
    """Compute radial basis functions of X, then fits a linear model between phi(X) and Y, and apply linear
    transformation to phi in order to compute non-linear approximation of Y

    Args:
        X: Input array
        Y: Dependent array, f(X)
        x_c: radial basis function centers
        ratio: float used to compute epsilon
    Returns:
        npt.NDArray[np.float64]: transformation of X according to linear model
    """

    epsilon = ratio * diameter(X)
    phi_x = radial_basis_function(X, x_c, epsilon)
    return linear_fit_transform(phi_x, Y)


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared Euclidean distance between two matrices.

    Args:
        y_true (np.ndarray): The matrix of actual values.
        y_pred (np.ndarray): The matrix of predicted values.

    Returns:
        float: The mean squared Euclidean distance.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred must be the same.")

    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2

    # Sum the squared differences row-wise
    row_squared_distances = np.sum(squared_diff, axis=1)

    # Compute the mean of these distances
    mean_squared_distance = np.mean(row_squared_distances)

    return mean_squared_distance


def diameter(X):
    """
    Compute the diameter of dataset X, that is max(dist(Xi,Xj)) for all Xi, Xj in the dataset. Used to compute epsilon parameter.

    Args:
        X: dataset

    Returns:
        Maximum of pairwise distances within the dataset
    """

    pairwise_distances = cdist(X, X, "euclidean")

    return np.max(pairwise_distances)


def approx_non_linear_field(X, centers, ratio):
    # x_c = np.linspace(-3.5, 4.5, L).reshape((L, 1))
    # X = X.reshape((X.shape[0],1))
    eps = ratio * diameter(X)
    phi_x = radial_basis_function(X, centers, eps)
    # return transform(X, least_squares(phi_x, Y))
    return phi_x


def x1_estim(f, x_0, delta_T):
    """
    Compute estimate of future state x1 given x0, delta_T and f, derivative of flow operator to be integrated over
    time

    Args:
      x_0: array representing system state at t=0
      f: callable function, derivative of flow operator
      delta_T: span of time separating state x0 and state x1 to be approximated

    Returns:
      Approximation of future state x1 after delta_T
    """
    for i in range(len(x_0)):
        solve = solve_ivp(f, [0, delta_T], x_0[i, :], t_eval=[delta_T])
        if i == 0:
            x1_approx = solve.y.T
        else:
            x1_approx = np.row_stack((x1_approx, solve.y.T))
        # print(solve.y.T)
    return x1_approx


def PHI(X, L, ratio):
    """
    Compute radial basis function values, given L and epsilon.

    Args:
    - X: dataset

    Returns:s
    - Radial basis function values
    """
    x_c = np.linspace(-4.5, 4.5, L).reshape((L, 1))
    # X = X.reshape((X.shape[0], 1))
    eps = ratio * diameter(X)
    phi_x = radial_basis_function(X, x_c, eps)
    # return transform(X, least_squares(phi_x, Y))
    return phi_x


def built_int_interpolator(X, Y, eps):
    # X_augment = np.column_stack([X, np.ones(X.shape)])

    return RBFInterpolator(X, Y, kernel="gaussian", epsilon=eps)


def trajectory(callable_fun, x_0, T, t_eval):
    solve = solve_ivp(callable_fun, [0, T], x_0, t_eval=t_eval)

    return solve.y.T


## add plot functions ?
