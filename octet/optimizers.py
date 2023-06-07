import numpy as np

def momentum(average_error, error, beta = 0.9):
    """Momentum optimizer

    Args:
        average_error (numpy.array): the average error of the previous errors, should be same shape as error
        error (numpy.array): current output error
        beta (float, optional): weightage of the influence of the average error over the new error in the calculation of the new average error. Defaults to 0.9.

    Returns:
        numpy.array: the new average error.
    """
    return beta * average_error + (1 - beta) * error

def no_optimizer(error):
    """No optimizer function

    Args:
        error (numpy.array): current output error

    Returns:
        numpy.array: current output error
    """
    return error

def rmsprop(s_prev, error, beta = 0.9, epsilon = 1e-8):
    """Root mean square propagation optimizer

    Args:
        s_prev (numpy.array): the previous value of s
        error (numpy.array): the current error
        beta (float, optional): weightage of the old value of s in the new computation of s. Defaults to 0.9.
        epsilon (float, optional): additive to the new value of s before squarerooting in order to prevent a division by 0. Defaults to 1e-8.

    Returns:
        numpy.array, numpy.array: a tuple of the new value of s and the error to be used in back propagation
    """
    s_new = beta * s_prev + (1 - beta) * (error ** 2)
    new_error = error / np.sqrt(s_new + epsilon)
    return s_new, new_error

def adam(s_prev, average_error, error, beta = 0.9, epsilon = 1e-8):
    """Adaptive momentum optimizer

    Args:
        s_prev (numpy.array): the previous value of s
        average_error (numpy.array): the previous average error
        error (numpy.array): the current error
        beta (float, optional): weightage given to the old s and average errors in the calculations of the new s and average error. Defaults to 0.9.
        epsilon (float, optional): additive to the new value of s before squarerooting to prevend a division by 0. Defaults to 1e-8.

    Returns:
        numpy.array, numpy.array, numpy.array: the new value of s, the new average error, the error to be used in the back propagation
    """
    s_new = beta * s_prev + (1 - beta) * (error ** 2)
    new_average_error = beta * average_error + (1 - beta) * error
    new_error = new_average_error / np.sqrt(s_new + epsilon)
    return s_new, new_average_error, new_error