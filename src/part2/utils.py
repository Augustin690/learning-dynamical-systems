import numpy as np
from scipy.spatial.distance import cdist
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
        delayed_embed = np.column_stack((data[0:-delta_t, col], data[delta_t:np.shape(data)[0] , col]))
        print('coucs')
        return delayed_embed

    elif out_dim == 3:
        delayed_embed = np.column_stack((data[0:-2*delta_t, col], data[delta_t: -delta_t, col],
                                         data[2*delta_t:np.shape(data)[0],col]))
        print('3d')
        return delayed_embed