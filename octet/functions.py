"""Contains activation functions and optimizer functions
"""

import numpy as np
from scipy.special import expit

# activation functions

def relu(input, derivative):
    """Rectified Linear Unit

    Args:
        input (numpy.array): a numpy array of numbers.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: rectified input or derivtive of rectified linear unit function with respect to the input.
    """
    if derivative:
        return 1 * np.greater_equal(input, 0)
    return np.maximum(input, 0)



def leaky_relu(input, derivative, leak = 0.001):
    """Leaky Rectified Linear Unit

    Args:
        input (numpy.array): numpy array of numbers.
        leak (float, optional): gradient of the function at negative inputs. Defaults to 0.001.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array : leaky relu of the input or the derivative of the leaky relu with respect to the input.
    """
    if derivative:
        return 1 * np.greater_equal(input, 0) + leak * (1 * np.less(input, 0))
    return input * np.greater(input, 0) + leak * (input * np.less(input, 0))

def parametric_relu6(input, derivative, param = 6):
    """Relu with a peak value

    Args:
        input (numpy.array): a numpy array of the input.
        param (int, optional): a maximum value of the function output. Defaults to 6.
        derivative (bool, optional): select whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: relu with a peak value given by param or the derivative with respect to the input.
    """
    if derivative:
        return 1 * np.bitwise_and(np.greater_equal(input, 0), np.less_equal(input, param))
    return input * np.bitwise_and(np.greater_equal(input, 0), np.less_equal(input, param))

def parametric_leaky_relu6(input, derivative, param = 6, leak = 0.001):
    """Leaky relu applied to both the maximum and minimum end of the input

    Args:
        input (numpy.array): a numpy array of the input.
        param (int, optional): the maximum point beyond which the function has a slope given by leak. Defaults to 6.
        leak (float, optional): the slope of the function beyond the value given by param and below 0. Defaults to 0.001.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: leaky relu version with leak at a maximum also or the derivative with respect to the input.
    """
    if derivative:
        return np.maximum(np.bitwise_and(1, np.bitwise_and(np.greater_equal(input, 0), np.less_equal(input, param))),leak)
    tmp = np.bitwise_and(np.greater_equal(input, 0), np.less_equal(input, param))
    return input * tmp +  leak * (input * np.bitwise_not(tmp))

def sigmoid(input, derivative):
    """Sigmoid function

    Args:
        input (numpy.array): numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: sigmoid function of the input or derivative with respect to the input.
    """
    if derivative:
        temp = sigmoid(input, False)
        return temp * (1 - temp)
    return (expit(input))

def tanh(input, derivative):
    """Hyperbolic tangent function

    Args:
        input (numpy.array): numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: hyperbolic tangent of the input or derivative with respect to the input.
    """
    if derivative:
        return 1 - np.tanh(input) ** 2
    return np.tanh(input)

def linear(input, derivative):
    """Linear activation function. Equivalent to no activation.

    Args:
        input (numpy.array): a numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: returns the input or derivative with respect to the input.
    """
    if derivative:
        return np.ones(input.shape)
    return input

def swish(input, derivative):
    """Swish function: similar to relu and sigmoid. Equivalent to the input multiplied by the sigmoid of the input.

    Args:
        input (numpy.array): a numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: returns the swish of the input or its derivative with respect to the input.
    """
    if derivative:
        tmp = 1 / (1 - np.exp(-input))
        return tmp + input * (tmp - tmp**2)
    return input / (1 + np.exp(-input))

def softmax(input, derivative):
    """Softmax: represents the output as a probability distribution.

    Args:
        input (numpy.array): a numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: returns a probability distribution based on the input or its derivative with respect to the input.
    """
    if derivative:
        m = np.tile(input, input.size)
        return (m.T * (np.identity(input.size) - m))
    temp = np.exp((input - np.max(input)))
    return temp / np.sum(temp)

def softmax2(input, derivative):
    """Softmax: represents the output as a probability distribution.

    Args:
        input (numpy.array): a numpy array of the input.
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: returns a probability distribution based on the input or its derivative with respect to the input.
    """
    if derivative:
        m = np.tile(input, input.size)
        return (m.T * (np.identity(input.size) - m))
    temp = np.exp((input - np.max(input)))
    return temp / np.sum(temp)

def elu(input, derivative):
    """Exponential Linear Unit

    Args:
        input (numpy.array): numpy array of the input
        derivative (bool, optional): selects whether the function returns the derivative with respect to the input or not. Defaults to False.

    Returns:
        numpy.array: returns the exponential of the input - 1 for values less than 0 and the input for values greater than 0 or the derivative with respect to the input.
    """
    tmp = np.greater(input, 0)
    if derivative:
        return 1 * tmp + np.bitwise_not(tmp) * np.exp(input)
    return input * tmp + np.bitwise_not(tmp) * (np.exp(input) - 1)