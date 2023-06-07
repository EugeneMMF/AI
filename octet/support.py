"""Contains supporting functions for the neural network
"""

from inspect import getfullargspec
from octet.functions import *
from octet.optimizers import *
# from functions import *
# from optimizers import *

def get_keywords(function):
    """Gets the keywords of a function

    Args:
        function (funct): the function whose keywords are required.

    Returns:
        list: keywords of the function provided.
    """
    args,varargs,varkw,defaults,_,_,_ = getfullargspec(function)
    if not defaults:
        return []
    return args[-len(defaults):]

def verify_activation(function, parameters):
    """Verifies that all the keywords required by the function are provided.

    Args:
        function (funct): the function.
        parameters (list): parameters provided to the function.

    Raises:
        Exception: parameters provided are not the same as the keywords of the function.
    """
    params = get_keywords(function)
    if (len(set(parameters.keys()).difference(set(params))) != 0) or (len(set(params).difference(set(parameters.keys()))) != 0):
        raise Exception("Missing parameters for activation function provided. Expected",params)