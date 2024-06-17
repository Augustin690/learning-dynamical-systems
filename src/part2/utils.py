import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


# TODO: Define all the functions (in addition to the ones below) required to complete tasks 4-5 in this python file.

def time_delay(data, col: int, delta_t: int, out_dim: int):
    """
    Create time delay coordinates from the data.
    :param col: The column/dimension no. of the data array to use.
    :param delta_t: No. of time-steps used for delayed measurement.
    :param out_dim: The target output dimension (2 or 3) for the output coordinates.
    :return: A numpy array of time-delay coordinates.
    
    Hint: 
    For out_dim = 2, 
        - shape of delayed coordinates = (np.shape(data)[0] - delta_t, 2)
        - Column 1: data with rows starting from 0 to -delta_t, col = col
        - Column 2: data with rows starting from delta_t to the last row, col = col
        - Stack the 2 columns to form the delayed coordinates

    For out_dim = 3, 
        - Column 1: data with rows starting from 0 to -2*delta_t, col = col
        - Column 2: data with rows starting from delta_t to -delta_t, col = col
        - Column 3: data with rows starting from 2*delta_t till the end, col = col
        - Stack the 3 columns to form the delayed coordinates    
    """
    # TODO: Implement this function!
    # column = data[:,col]
    if out_dim == 2:
        # delayed_embed = np.empty((np.shape(data)[0] - delta_t), out_dim)
        delayed_embed = np.column_stack((data[0:-delta_t, col], data[delta_t:np.shape(data)[0], col]))
        return delayed_embed

    elif out_dim == 3:
        delayed_embed = np.column_stack((data[0:-2 * delta_t, col], data[delta_t: -delta_t, col],
                                         data[2 * delta_t:np.shape(data)[0], col]))

        return delayed_embed


# Function to create delay embedding windows
def create_windows(data, num_delays, num_areas):
    windows = []
    for start in range(len(data) - num_delays):
        window = data[start:start + num_delays + 1, :num_areas]
        windows.append(window.flatten())
    return np.array(windows)


def center_data(data: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """ Center data by subtracting the mean of the data.

    Args:
        data (npt.NDArray[np.float64]): Data matrix.

    Returns:
        npt.NDArray[np.float64]: centered data.
    """
    # TODO: Implement method
    mean = np.mean(data, axis=0)
    return data - mean


def compute_svd(data: np.ndarray[np.float64]) -> tuple[
    np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
    """ Compute (reduced) SVD of the data matrix. Set (full_matrices=False).

    Args:
        data (npt.NDArray[np.float]): data matrix.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: U, S, V_t.
    """

    # TODO: Implement method
    # u, s, v = np.linalg.svd(data)
    return np.linalg.svd(data, full_matrices=False)


def compute_energy(S: np.ndarray[np.float64], c:int = 1) -> np.float64:
    """
    Percentage of total “energy” (explained variance) of (only) the i-th principal component of singular value on the diagonal of the matrix S.
        Note that it is NOT a sum of first 'c' components!

    Args:
        S (npt.NDArray[np.float64]): Array containing the singular values of the data matrix
        c (int): Component of SVD (Starts from 1, NOT 0). E.g set c = 1 for first component. Defaults to 1.

    Returns:
        np.float64: percentage energy in the c-th principal component
    """
    # TODO: Implement method.
    total_energy = np.sum(S**2)
    energy = S**2 / total_energy
    return energy[c-1] * 100

### FUNCTIONS FROM PART 1 USED TO APPROXIMATE LORENZ VECTOR FIELD


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
    phi_x = np.exp(-(cdist(x_c, x, 'sqeuclid') / eps ** 2))

    return phi_x


def diameter(X):
    # return np.max(X) - np.min(X)
    pairwise_distances = cdist(X, X, 'euclidean')
    # Find the maximum distance
    return np.max(pairwise_distances)


def approx_non_linear_field(X, centers, ratio):
    # x_c = np.linspace(-3.5, 4.5, L).reshape((L, 1))
    # X = X.reshape((X.shape[0],1))
    eps = ratio * diameter(X)
    phi_x = radial_basis_function(X, centers, eps)
    # return transform(X, least_squares(phi_x, Y))
    return phi_x


def least_squares(A, b, cond=0.001):
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
    # A_augment = np.column_stack([A, np.ones(A.shape)])
    x, residuals, rank, s = lstsq(A, b, cond=cond)

    # Return the solution
    return x


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
