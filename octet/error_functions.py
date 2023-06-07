import numpy as np

# error functions#
# mean squared error
def mse(y_true, y_pred, **kwargs):
    if kwargs.get('derivative'):
        return 2 * (y_pred - y_true)
    return np.mean((y_true - y_pred)**2)

# mean quadrupled error
def mqe(y_true, y_pred, **kwargs):
    if kwargs.get('derivative'):
        return 4 * (y_pred - y_true)**3
    return np.mean((y_true - y_pred)**4)

# categorical cross entropy
def cce(y_true, y_pred, **kwargs):
    if kwargs.get('derivative'):
        return -(y_true / (y_pred + 1e-8))
    return -np.mean(y_true * np.log(y_pred + 1e-8))

# binary cross entropy
def bce(y_true, y_pred, **kwargs):
    if kwargs.get('derivative'):
        return -(y_true / (y_pred + 1e-4) - (1 - y_true) / (1 - y_pred + 1e-4))
    return -np.mean(y_true * np.log(y_pred + 1e-4) + (1 - y_true) * np.log(1 - y_pred + 1e-4))