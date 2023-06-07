"""Sub functions for support.
"""

from inspect import isfunction, getmembers, signature, getargs, getargspec
import octet.functions as functions
import octet.optimizers as optimizers
import octet.error_functions as error_functions
# import functions
# import error_functions
# import error_functions

def denumpy(input):
    """Converts a numpy array into a python list.

    Args:
        input (numpy.array): the numpy array to be converted to a python list.

    Returns:
        list: a python list of the same dimensions as the input.
    """
    try:
        len(input)
        return list(denumpy(sub) for sub in input)
    except:
        return input

def get_function(function_name):
    """Get the function based on the name provided from the functions module

    Args:
        function_name (str): the name of the function desired

    Raises:
        Exception: if the no function in the functions module has the same name

    Returns:
        function: the function
    """
    if function_name == None:
        return functions.linear
    all_fucntions = getmembers(functions, isfunction)
    for funct in all_fucntions:
        if funct[0] == function_name:
            return funct[1]#, getargspec(funct[1])[0]
    raise Exception("Function not defined")

def get_optimizer(optimizer_name):
    """Get the function based on the name provided from the optimizers module

    Args:
        optimizer_name (str): the name of the optimizer desired

    Raises:
        Exception: if the no optimizer in the optimizers module has the same name

    Returns:
        function: the optimizer
    """
    if optimizer_name == None:
        return optimizers.no_optimizer
    all_optimizers = getmembers(optimizers, isfunction)
    for opt in all_optimizers:
        if opt[0] == optimizer_name:
            return opt[1]#, getargspec(funct[1])[0]
    raise Exception("Optimizer not defined")

def get_error_function(function_name):
    """Get the function based on the name provided from the error_functions module

    Args:
        function_name (str): the name of the error function desired

    Raises:
        Exception: if the no error function in the error_functions module has the same name

    Returns:
        function: the error function
    """
    all_fucntions = getmembers(error_functions, isfunction)
    for funct in all_fucntions:
        if funct[0] == function_name:
            return funct[1]#, getargspec(funct[1])[0]
    raise Exception("Error function not defined")