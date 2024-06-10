import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import pandas as pd
from scipy.spatial.distance import cdist

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
    # Hint: Use cdist from scipy.spatial.distance 
    pass


def least_squares(A, b, cond=1e-5):
    """ Returns a Least squares solution of Ax = b

    Args:
        A (npt.NDArray[np.float32]): Input array for the least squres problem
        b (npt.NDArray[np.float32]): Output array for the least squares problem
    Returns:
        npt.NDArray[np.float32]: Least squares solution of Ax = b
    """
    # TODO: Implement using scipy.linalg.lstsq 
    # Hint: Don't forget that lstsq also returns other parameters such as residuals, rank, singular values etc. 
    pass 
    

