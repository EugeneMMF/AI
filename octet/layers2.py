"""This module contains the following layers available for use in a neural network:
1. Input
2. Dense
3. Conv2D
4. Flatten
5. MaxPool2D
6. AveragePool2D
7. SimpleRNN
8. LSTM
9. Dropout
10. LSTM2
11. RepeatVector
12. TimeDistributed
13. Attention
14. Activation
15. Reshape
"""
import numpy as np
from octet.subfunctions import get_function, denumpy, get_optimizer
from octet.support import *
from scipy import signal
# from keras.layers import *
# from subfunctions import get_function, denumpy, get_optimizer
# from support import *

class BaseLayer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def compile(self, input_shape):
        raise NotImplementedError
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, error, learning_rate, optimizer = None, parameters = {}):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def load(self, **data):
        raise NotImplementedError
    
    def reset_optimizer(self):
        raise NotImplementedError

class Input(BaseLayer):
    def __init__(self, input_shape, activation = None, parameters = {}):
        """Input layer

        Args:
            input_shape (tuple): tuple of the dimensions of the input
            activation (str): string of the name of the activation function to be used on the output
            param (int or float): a number the activaion function output is multiplied by in parametric activation functions
            leak (int or float): a number describing the amount of leakage in leaky activation functions or a number by which functions inputs are divided by in parametric activation functions
        """
        self.input_shape = input_shape
        self.activation_name = activation
        self.parameters = parameters
    
    def compile(self, input_shape):
        """Initialise the layer properly

        Args:
            input_shape (tuple): a tuple representing the shape the layer accepts as its input

        Returns:
            tuple: a tuple representing the shape of the output of the layer
        """
        self.activation = get_function(self.activation_name)
        verify_activation(self.activation, self.parameters)
        return self.input_shape
    
    def forward(self, input):
        """forward propagation method

        Args:
            input (numpy.array): an numpy array representation of the input to the layer

        Returns:
            numpy.array: a numpy array of the output of the layer
        """
        self.input = input
        self.output = self.activation(input, derivative = False, **self.parameters)
        return self.output
    
    def backward(self, error, learning_rate, optimizer = None, parameters = {}):
        """backward propagation

        Args:
            error (numpy.array): the error of in the output. Must be of the same shape as the output of the layer.
            learning_rate (float): a number that determines how much the error affects the change in the learable parameters.
            optimizer (function, optional): an optimizer function defined in functions to improve learning. Defaults to None.
            parameters (dict, optional): parameters associated with the optimizer function specified. Defaults to {}.

        Returns:
            numpy.array: input error of the layer
        """
        if self.activation_name == "softmax":
            error = np.dot(self.activation(self.output, derivative = True, **self.parameters), error)
        else:
            error *= self.activation(self.input, derivative = True, **self.parameters)
        return error
    
    def save(self):
        """For saving the layer

        Returns:
            str: string of the parameters needed to reinitialise the layer in a new model
        """
        return f"Input\n{str(self.input_shape)}\n{str(self.activation_name)}\n{str(self.parameters)}\n"
    
    def load(self, **data):
        """For initialising the layer parameters based on known data
        
        Args:
            activation (str): a string of the activation function of the layer
        """
        self.activation_name = data['activation']
        self.activation = get_function(self.activation_name)
        self.parameters = data['parameters']
        verify_activation(self.activation, self.parameters)
    
    def reset_optimizer(self):
        pass

class Dense(BaseLayer):
    def __init__(self, nodes, input_shape = None, activation = None, parameters = {}):
        """Creates a layer of artificial neurons fully connected to the next and previous layers.

        Args:
            nodes (int): number of artificial neurons in the layer. Output shape is nodes x 1.
            input_shape (tuple, optional): shape of the input. Shape is R x C. Defaults to None.
            activation (string, optional): activation name of the function to be applied to the output. Defaults to None.
            parameters (dict, optional): parameters to be applied to the activation functions and learning parameters. Defaults to {}.
        """
        self.nodes = nodes
        self.input_shape = input_shape
        self.output_shape = (nodes,)
        self.activated_output = np.zeros(self.output_shape)
        self.activation_name = activation
        self.activation = None
        self.parameters = parameters
        self.biases = np.zeros((self.output_shape))
        self.activation = get_function(activation)
        verify_activation(self.activation,parameters)
        if input_shape:
            self.weights = np.random.uniform(-1./np.sqrt(self.input_shape[0]), 1./np.sqrt(self.input_shape[0]), (input_shape[0],self.nodes))
        else:
            self.weights = None
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
    
    def compile(self, input_shape):
        """Initialise the layer.

        Args:
            input_shape (tuple): shape of the input to the layer.

        Raises:
            Exception: input shape provided does not match the input shape used in the definition of the layer.

        Returns:
            tuple: shape of the output of the layer.
        """
        if self.input_shape == None:
            self.input_shape = input_shape
        if self.input_shape != input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        self.weights = np.random.uniform(-1./np.sqrt(self.input_shape[0]), 1./np.sqrt(self.input_shape[0]), (input_shape[0],self.nodes))
        self.activation = get_function(self.activation_name)
        return self.output_shape
    
    def forward(self, input):
        """Forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer in the same shape as the input shape defined when creating the layer.

        Returns:
            numpy.array: the output of forward propagation after applying an activation function defined.
        """
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        self.activated_output = self.activation(self.output,derivative = False,**self.parameters)
        return self.activated_output
    
    def backward(self, error, learning_rate, optimizer = None, parameters = {}):
        """Backward propagation of the layer.

        Args:
            error (numpy.array): the error of in the output. Must be of the same shape as the output of the layer.
            learning_rate (float): a number that determines how much the error affects the change in the learable parameters.
            optimizer (function, optional): an optimizer function defined in functions to improve learning. Defaults to None.
            parameters (dict, optional): parameters associated with the optimizer function specified. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not a valid optimizer function defined in functions.

        Returns:
            numpy.array: the error associated with the input to the layer in the same shape as the layer input.
        """
        if self.activation_name == "softmax2":
            error = np.dot(self.activation(self.activated_output,derivative = True, **self.parameters), error)
        else:
            error *= self.activation(self.output, derivative = True, **self.parameters)
        unaltered_error = np.copy(error)
        if optimizer == momentum:
            self.average_error = momentum(self.average_error, error, **parameters)
            error = self.average_error
        elif optimizer == rmsprop:
            self.s_value, error = rmsprop(self.s_value, error, **parameters)
        elif optimizer == adam:
            self.s_value, self.average_error, error = adam(self.s_value, self.average_error, error, **parameters)
        elif optimizer == None or no_optimizer:
            pass
        else:
            raise Exception("Invalid optimizer")
        self.input_error = np.dot(unaltered_error, self.weights.T)
        self.weights -= learning_rate * np.dot(self.input.T, error)
        self.biases -= learning_rate * error
        return self.input_error
    
    def get_errors(self, error,optimizer = None, parameters = {}):
        if self.activation_name == "softmax":
            error = np.dot(self.activation(self.activated_output,derivative = True, **self.parameters), error)
        else:
            error *= self.activation(self.output, derivative = True, **self.parameters)
        unaltered_error = np.copy(error)
        if optimizer == momentum:
            self.average_error = momentum(self.average_error, error, **parameters)
            error = self.average_error
        elif optimizer == rmsprop:
            self.s_value, error = rmsprop(self.s_value, error, **parameters)
        elif optimizer == adam:
            self.s_value, self.average_error, error = adam(self.s_value, self.average_error, error, **parameters)
        elif optimizer == None or no_optimizer:
            pass
        else:
            raise Exception("Invalid optimizer")
        input_error = np.dot(unaltered_error, self.weights.T)
        weights_error = np.dot(self.input.T, error)
        biases_error = error
        return input_error, weights_error, biases_error
    
    def update_learnable_parameters(self, weights_error, biases_error, learning_rate):
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biases_error
    
    def save(self):
        return f"Dense\n{str(self.nodes)}\n{str(self.input_shape)}\n{str(self.output_shape)}\n{str(self.weights.shape)}\n{str(self.activation_name)}\n{str(self.parameters)}\n{str(denumpy(self.weights))}\n{str(denumpy(self.biases))}\n"
    
    def load(self, **data):
        self.activation_name = data['activation']
        self.activation = get_function(self.activation_name)
        self.parameters = data['parameters']
        verify_activation(self.activation, self.parameters)
        self.input_shape = data['input_shape']
        self.output_shape = data['output_shape']
        self.biases = data['biases']
        self.weights = data['weights']
    
    def reset_optimizer(self):
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
    
class Conv2D(BaseLayer):
    def __init__(self, filters, kernel_size, input_shape = None, strides = (1,1), padding = "valid", activation = None, parameters = {}):
        """Creates a 2D convolutional layer to perform convolution on a 2D input.

        Args:
            filters (int): number of filters to be applied to the input. Output shape is Ro x Co x filters
            kernel_size (tuple): tuple of the shape of the size of the kernel in terms of rows and columns.
            input_shape (tuple, optional): the shape of the input to the layer. Input shape is R x C x D. Defaults to None.
            strides (tuple, optional): the number of strides taken when sliding the kernel over the input. Defaults to (1,1).
            padding (str, optional): "valid" or "same". "valid" gives a valid convolution with no padding with zeros to the input. "same" adds padding to give an output shape that is the same as the input shape. Defaults to "valid".
            activation (string, optional): an activation function name to be applied to the output of the convolution. Defaults to None.
            parameters (dict, optional): parameters associated with the activation function defined. Defaults to {}.

        Raises:
            Exception: if softmax activation is used.
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        if activation == "softmax":
            raise Exception("Softmax can only be applied for dense layers")
        self.activation_name = activation
        self.parameters = parameters
        self.input_shape = input_shape
    
    def compile(self, input_shape):
        """Initialise the layer.

        Args:
            input_shape (tuple): shape of the input to the layer.

        Raises:
            Exception: if the input shape passed to the function does not match the input shape defined when creating the layer.
            Exception: if the stride chosen leads to some part of the kernel not overlapping with the input.

        Returns:
            tuple: shape of the output of the layer.
        """
        self.activation = get_function(self.activation_name)
        verify_activation(self.activation, self.parameters)
        if not self.input_shape and input_shape != self.input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        self.input_shape = input_shape
        if self.padding == "same":
            self.output_shape = self.input_shape
        else:
            self.output_shape = ((self.input[0] - self.kernel_size[0])/self.strides[0] + 1, (self.input_shape[1] - self.kernel_size[1])/self.strides[1] + 1, self.filters)
            if self.output_shape != (int(self.output_shape[0]), int(self.output_shape[1]), int(self.output_shape[2])):
                raise Exception("The combination of stride, input_shape and kernel_size is invalid. Output is not perfect.")
        self.kernel_shape = (self.filters, self.input_shape[2], self.kernel_size[0], self.kernel_size[1])
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(*self.output_shape)
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
        return self.output_shape
    
    def forward(self, input):
        """Perform forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer.

        Returns:
            numpy.array: the output of the layer.
        """
        self.input = input
        output = np.copy(self.biases)
        for i in range(self.output_shape[2]):
            for j in range(self.input_shape[2]):
                output[:,:,i] += signal.correlate2d(self.input[:,:,j], self.kernels[i,j], "valid")
        self.output = output
        self.activated_output = self.activation(output, derivative=False, **self.parameters)
        return self.activated_output
    
    def backward(self, error, learning_rate, optimizer = None, parameters = {}):
        """Perform backward propagation of the layer and update the learnable parameters of the layer.

        Args:
            error (numpy.array): the error associated with the output of the layer. Must be of the same shape as the output of the layer.
            learning_rate (float): the amount by which the error affects the learnable parameters.
            optimizer (function, optional): optimization function to be used when training. Defaults to None.
            parameters (dict, optional): parameters associated with the optimization function. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not defined.

        Returns:
            numpy.array: the error associated with the input of the layer.
        """
        error *= self.activation(self.output, derivative = True, **self.parameters)
        unaltered_error = np.copy(error)
        if optimizer == momentum:
            self.average_error = momentum(self.average_error, error, **parameters)
            error = self.average_error
        elif optimizer == rmsprop:
            self.s_value, error = rmsprop(self.s_value, error, **parameters)
        elif optimizer == adam:
            self.s_value, self.average_error, error = adam(self.s_value, self.average_error, error, **parameters)
        elif optimizer == None or no_optimizer:
            pass
        else:
            raise Exception("Invalid optimizer")
        kernels_error = np.zeros(self.kernels.shape)
        input_error = np.zeros(self.input_shape)
        biases_error = np.copy(error)
        for i in range(self.input_shape[2]):
            for j in range(self.filters):
                kernels_error[j][i] = signal.correlate2d(self.input[:,:,i], error[:,:,j],"valid")
                input_error[:,:,i] += signal.convolve2d(self.kernels[j][i], unaltered_error[:,:,j],"full")
        self.kernels -= learning_rate * kernels_error
        self.biases -= learning_rate * biases_error
        return input_error
    
    def save(self):
        return f"Conv2D\n{str(self.filters)}\n{str(self.filter_shape)}\n{str(self.kernel_size)}\n{str(self.input_shape)}\n{str(self.output_shape)}\n{str(self.kernels.shape)}\n{str(self.padding)}\n{str(self.strides)}\n{str(self.activation_name)}\n{str(self.parameters)}\n{str(denumpy(self.kernels))}\n{str(denumpy(self.biases))}\n"

    def load(self, **data):
        self.filters = data['filters']
        self.filter_shape = data['filter_shape']
        self.input_shape = data['input_shape']
        self.output_shape = data['output_shape']
        self.padding = data['padding']
        self.strides = data['strides']
        self.activation_name = data['activation']
        self.kernels = data['kernels']
        self.biases = data['biases']
        self.parameters = data['parameters']
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
        if self.activation_name == "softmax":
            raise Exception("Softmax cannot be used in Conv2D layers")
        self.activation = get_function(self.activation_name)
        verify_activation(self.activation, self.parameters)

    def reset_optimizer(self):
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)

class Flatten(BaseLayer):
    def __init__(self):
        pass
    
    def compile(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.product(input_shape),)
        return self.output_shape
    
    def forward(self, input):
        """Perform forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer. Must be a sequence of inputs.

        Returns:
            numpy.array: the output of the layer.
        """
        self.input = input
        self.output = np.reshape(input, self.output_shape)
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        """Perform backward propagation of the layer and update the learnable parameters of the layer.

        Args:
            error (numpy.array): the error associated with the output of the layer. Must be of the same shape as the output of the layer.
            learning_rate (float): the amount by which the error affects the learnable parameters.
            optimizer (function, optional): optimization function to be used when training. Defaults to None.
            parameters (dict, optional): parameters associated with the optimization function. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not defined.

        Returns:
            numpy.array: the error associated with the input of the layer.
        """
        return np.reshape(error, self.input_shape)
    
    def save(self):
        return f"Flatten\n{str(self.input_shape)}\n{str(self.output_shape)}\n"
    
    def load(self, **data):
        self.input_shape = data['input_shape']
        self.output_shape = data['output_shape']
    
    def reset_optimizer(self):
        pass
    
class MaxPool2D(BaseLayer):
    def __init__(self, pool_size):
        _,_ = pool_size
        self.pool_size = pool_size
    
    def compile(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / self.pool_size[0]), int(input_shape[1] / self.pool_size[1]), input_shape[2])
        return self.output_shape

    def forward(self, input):
        """Perform forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer. Must be a sequence of inputs.

        Returns:
            numpy.array: the output of the layer.
        """
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(self.output_shape[2]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[0]):
                    h = k * self.pool_size[0]
                    w = j * self.pool_size[1]
                    self.output[k,j,i] = np.max(input[h : h + self.pool_size[0], w : w + self.pool_size[1], i])
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        """Perform backward propagation of the layer and update the learnable parameters of the layer.

        Args:
            error (numpy.array): the error associated with the output of the layer. Must be of the same shape as the output of the layer.
            learning_rate (float): the amount by which the error affects the learnable parameters.
            optimizer (function, optional): optimization function to be used when training. Defaults to None.
            parameters (dict, optional): parameters associated with the optimization function. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not defined.

        Returns:
            numpy.array: the error associated with the input of the layer.
        """
        input_error = np.zeros(self.input_shape)
        for i in range(self.input_shape[2]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[0]):
                    input_error[k,j,i] = error[int(k / self.pool_size[0]), int(j / self.pool_size[1]), i]
        return input_error
    
    def save(self):
        return f"MaxPool2D\n{str(self.pool_size)}\n{str(self.input_shape)}\n{str(self.output_shape)}\n"
    
    def load(self, **data):
        self.input_shape = data['input_shape']
        self.output_shape = data['output_shape']
        
    def reset_optimizer(self):
        pass
    
class AveragePool2D(BaseLayer):
    def __init__(self, pool_size):
        _,_ = pool_size
        self.pool_size = pool_size
    
    def compile(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / self.pool_size[0]), int(input_shape[1] / self.pool_size[1]), input_shape[2])
        return self.output_shape

    def forward(self, input):
        """Perform forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer. Must be a sequence of inputs.

        Returns:
            numpy.array: the output of the layer.
        """
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(self.output_shape[2]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[0]):
                    h = k * self.pool_size[0]
                    w = j * self.pool_size[1]
                    self.output[k,j,i] = np.mean(input[h : h + self.pool_size[0], w : w + self.pool_size[1], i])
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        """Perform backward propagation of the layer and update the learnable parameters of the layer.

        Args:
            error (numpy.array): the error associated with the output of the layer. Must be of the same shape as the output of the layer.
            learning_rate (float): the amount by which the error affects the learnable parameters.
            optimizer (function, optional): optimization function to be used when training. Defaults to None.
            parameters (dict, optional): parameters associated with the optimization function. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not defined.

        Returns:
            numpy.array: the error associated with the input of the layer.
        """
        input_error = np.zeros(self.input_shape)
        for i in range(self.input_shape[2]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[0]):
                    input_error[k,j,i] = error[int(k / self.pool_size[0]), int(j / self.pool_size[1]), i]
        return input_error
    
    def save(self):
        return f"AveragePool2D\n{str(self.pool_size)}\n{str(self.input_shape)}\n{str(self.output_shape)}\n"
    
    def load(self, **data):
        self.input_shape = data['input_shape']
        self.output_shape = data['output_shape']
    
    def reset_optimizer(self):
        pass
        
class SimpleRNN(BaseLayer):
    def __init__(self, units, input_shape = None, activation = "tanh", parameters = {}):
        """Create a simple recurrent neural network unit.

        Args:
            units (int): the number of neurons in the layer.
            input_shape (tuple, optional): the shape of the input to the layer. Note that it must be of the form length of the input sequence x R x C. Defaults to None.
            activation (str, optional): the name of the activation function to be used in the hidden layers. Defaults to "tanh".
            parameters (dict, optional): the paramters associated with the activation function defined earlier. Defaults to {}.
        """
        if input_shape != None:
            self.sequence_length, *self.x_shape = self.input_shape = input_shape
            self.x_shape = tuple(self.x_shape)
        self.activation_name = activation
        self.parameters = parameters
        self.units = units
    
    def compile(self, input_shape):
        """Inititalise the layer properly.

        Args:
            input_shape (tuple): shape of the input to the layer.

        Raises:
            Exception: if the input shape defined when creating the layer is not the same as the shape of the input from the previous layer.

        Returns:
            tuple: the shape of the output of the layer
        """
        if not self.input_shape and input_shape != self.input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        self.input_shape = input_shape
        self.activation = get_function(self.activation_name)
        verify_activation(self.activation, self.parameters)
        self.sequence_length, *self.x_shape = self.input_shape = input_shape
        self.x_shape = tuple(self.x_shape)
        self.output_shape = self.x_shape
        self.input_weights = np.random.uniform(-np.sqrt(1./self.x_shape[0]), np.sqrt(1./self.x_shape[0]), (self.units, self.x_shape[0]))
        self.output_weights = np.random.uniform(-np.sqrt(1./self.units), np.sqrt(1./self.units), (self.x_shape[0], self.units))
        self.hidden_weights = np.random.uniform(-np.sqrt(1./self.units), np.sqrt(1./self.units), (self.units, self.units))
        self.hidden_biases = np.zeros((self.units, 1))
        self.output_biases = np.zeros(self.x_shape)
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
        return self.output_shape
    
    def forward(self, input):
        """Perform forward propagation of the layer.

        Args:
            input (numpy.array): the input to the layer. Must be a sequence of inputs.

        Returns:
            numpy.array: the output of the layer.
        """
        self.input = input
        self.hidden_outputs = np.zeros((self.sequence_length, self.units, 1))
        self.activated_hidden_outputs = np.zeros_like(self.hidden_outputs)
        self.output = np.copy(self.output_biases)
        self.hidden_outputs[-1] = np.zeros_like(self.hidden_biases)
        for i in range(self.sequence_length):
            self.hidden_outputs[i] = np.copy(self.hidden_biases)
            self.hidden_outputs[i] += np.dot(self.input_weights, input[i]) + np.dot(self.hidden_weights, self.hidden_outputs[i-1])
            self.activated_hidden_outputs[i] = self.activation(self.hidden_outputs[i], derivative=False, **self.parameters)
        self.output += np.dot(self.output_weights, self.activated_hidden_outputs[-1])
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        """Perform backward propagation of the layer and update the learnable parameters of the layer.

        Args:
            error (numpy.array): the error associated with the output of the layer. Must be of the same shape as the output of the layer.
            learning_rate (float): the amount by which the error affects the learnable parameters.
            optimizer (function, optional): optimization function to be used when training. Defaults to None.
            parameters (dict, optional): parameters associated with the optimization function. Defaults to {}.

        Raises:
            Exception: if the optimizer function is not defined.

        Returns:
            numpy.array: the error associated with the input of the layer.
        """
        if optimizer == momentum:
            self.average_error = momentum(self.average_error, error, **parameters)
            error = self.average_error
        elif optimizer == rmsprop:
            self.s_value, error = rmsprop(self.s_value, error, **parameters)
        elif optimizer == adam:
            self.s_value, self.average_error, error = adam(self.s_value, self.average_error, error, **parameters)
        elif optimizer == None or no_optimizer:
            pass
        else:
            raise Exception("Invalid optimizer")
        input_error = np.dot(self.output_weights.T, error)
        self.output_biases -= learning_rate * error
        self.output_weights -= learning_rate * np.dot(error, self.activated_hidden_outputs[-1].T)
        hidden_biases_error = np.zeros_like(input_error)
        hidden_weights_error = np.zeros_like(self.hidden_weights)
        input_weights_error = np.zeros_like(self.input_weights)
        for i in reversed(range(self.sequence_length)):
            error = np.copy(input_error)
            if self.activation_name == "softmax":
                error = np.dot(self.activation(self.activated_hidden_outputs[i], derivative = True, **self.parameters), error)
            else:
                error *= self.activation(self.hidden_outputs[i], derivative = True, **self.parameters)
            input_error = np.dot(self.hidden_weights.T, error)
            hidden_biases_error += error
            input_weights_error += np.dot(error, self.input[i].T)
            if i > 0:
                hidden_weights_error += np.dot(error, self.hidden_outputs[i-1].T)
        self.hidden_biases -= learning_rate * hidden_biases_error
        self.hidden_weights -= learning_rate * hidden_weights_error
        self.input_weights -= learning_rate * input_weights_error
        return input_error
    
    def save(self):
        return f"SimpleRNN\n{str(self.units)}\n{str(self.input_shape)}\n{self.activation_name}\n{str(self.parameters)}\n{str(self.hidden_biases)}\n{str(self.hidden_weights)}\n{str(self.input_weights)}\n{str(self.output_weights)}\n{str(self.output_biases)}\n"
    
    def load(self, **data):
        self.parameters = data['parameters']
        self.hidden_biases = data['hidden_biases']
        self.hidden_weights = data['hidden_weights']
        self.input_weights = data['input_weights']
        self.output_biases = data['output_biases']
        self.output_weights = data['output_weights']
    
    def reset_optimizer(self):
        self.average_error = np.zeros(self.output_shape)
        self.s_value = np.zeros(self.output_shape)
        
class LSTM(BaseLayer):
    def __init__(self, units, input_shape, output_sequence, final_activation, parameters = {}, return_sequence = False):
        self.units = units
        self.output_sequence = output_sequence
        self.final_activation_name = final_activation
        self.final_activation = get_function(final_activation)
        self.parameters = parameters
        verify_activation(self.final_activation, parameters)
        self.input_shape = input_shape
        self.sequence_length, *self.x_shape = input_shape
        self.x_shape = tuple(self.x_shape)
        self.return_sequence = return_sequence
        self.cell_state = np.zeros((self.sequence_length + self.output_sequence - 1, units, 1))
        self.hidden_state = np.zeros((self.sequence_length + self.output_sequence - 1, units, 1))
        # self.output_weights = np.random.rand(self.x_shape[0], units)
        # self.output_biases = np.zeros(self.x_shape)
        self.forget_gate = Dense(self.units, (self.x_shape[0] + units, 1), "linear")
        self.tanh_input_gate = Dense(self.units, (self.x_shape[0] + units, 1), "linear")
        self.sigmoid_input_gate = Dense(self.units, (self.x_shape[0] + units, 1), "linear")
        self.output_gate = Dense(self.units, (self.x_shape[0] + units, 1), "linear")
        self.final_output_dense = Dense(self.x_shape[0], (units, 1), "linear")
        self.forget_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.forget_gate_output = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.tanh_input_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.tanh_input_gate_output = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.sigmoid_input_gate_output = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.sigmoid_input_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.output_gate_output = np.zeros((self.sequence_length + self.output_sequence, units, 1))
        self.output_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, units, 1))
    
    def compile(self, input_shape):
        if self.input_shape != input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        if self.return_sequence:
            return self.hidden_state.shape
        return tuple((self.output_sequence,*self.x_shape))
    
    def forward(self, input):
        # self.forget_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.forget_gate_output = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.tanh_input_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.tanh_input_gate_output = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.sigmoid_input_gate_output = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.sigmoid_input_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.output_gate_output = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        # self.output_gate_output_activated = np.zeros((self.sequence_length + self.output_sequence, self.units, 1))
        self.cell_state = np.zeros((self.sequence_length + self.output_sequence - 1, self.units, 1))
        self.hidden_state = np.zeros((self.sequence_length + self.output_sequence - 1, self.units, 1))
        self.output = []
        self.unactivated_output = np.zeros((self.sequence_length + self.output_sequence - 1, *self.x_shape))
        self.input = np.zeros((self.sequence_length + self.output_sequence, *self.x_shape))
        self.input[:self.sequence_length] = input
        for i in range(self.sequence_length + self.output_sequence - 1):
            if i >= (self.sequence_length - 1):
                self.concatenated_input = np.concatenate((self.input[i], self.hidden_state[i - 1]))
                self.forget_gate_output[i] = self.forget_gate.forward(self.concatenated_input)
                self.forget_gate_output_activated[i] = sigmoid(self.forget_gate_output[i], False)
                self.tanh_input_gate_output[i] = self.tanh_input_gate.forward(self.concatenated_input)
                self.tanh_input_gate_output_activated[i] = tanh(self.tanh_input_gate_output[i], False)
                self.sigmoid_input_gate_output[i] = self.sigmoid_input_gate.forward(self.concatenated_input)
                self.sigmoid_input_gate_output_activated[i] = sigmoid(self.sigmoid_input_gate_output[i], False)
                self.output_gate_output[i] = self.output_gate.forward(self.concatenated_input)
                self.output_gate_output_activated[i] = sigmoid(self.output_gate_output[i], False)
                self.cell_state[i] = (self.cell_state[i - 1] * self.forget_gate_output_activated[i]) + (self.tanh_input_gate_output_activated[i] * self.sigmoid_input_gate_output_activated[i])
                self.hidden_state[i] = tanh(self.cell_state[i],False) * self.output_gate_output_activated[i]
                self.unactivated_output[i] = self.final_output_dense.forward(self.hidden_state[i])
                self.output.append(self.final_activation(self.unactivated_output[i], False, **self.parameters))
                # self.output.append(np.dot(self.output_weights, self.hidden_state[i]) + self.output_biases)
                self.input[i + 1] = self.output[-1]
            else:
                self.concatenated_input = np.concatenate((self.input[i], self.hidden_state[i - 1]))
                self.forget_gate_output[i] = self.forget_gate.forward(self.concatenated_input)
                self.forget_gate_output_activated[i] = sigmoid(self.forget_gate_output[i], False)
                self.tanh_input_gate_output[i] = self.tanh_input_gate.forward(self.concatenated_input)
                self.tanh_input_gate_output_activated[i] = tanh(self.tanh_input_gate_output[i], False)
                self.sigmoid_input_gate_output[i] = self.sigmoid_input_gate.forward(self.concatenated_input)
                self.sigmoid_input_gate_output_activated[i] = sigmoid(self.sigmoid_input_gate_output[i], False)
                self.output_gate_output[i] = self.output_gate.forward(self.concatenated_input)
                self.output_gate_output_activated[i] = sigmoid(self.output_gate_output[i], False)
                self.cell_state[i] = (self.cell_state[i - 1] * self.forget_gate_output_activated[i]) + (self.tanh_input_gate_output_activated[i] * self.sigmoid_input_gate_output_activated[i])
                self.hidden_state[i] = tanh(self.cell_state[i],False) * self.output_gate_output_activated[i]
        self.output = np.array(self.output)
        if self.return_sequence:
            return self.hidden_state
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={},previous_layer_return_sequence = False):
        def clip_error(error, limit):
            error = np.clip(error, -limit, limit)
            return error
        
        limit = 5e8
        output_error = np.zeros((self.sequence_length + self.output_sequence - 1, *self.x_shape))
        if not self.return_sequence:
            output_error[self.sequence_length - 1:] = np.copy(error)
        hidden_state_error = np.zeros_like(self.hidden_state[0])
        cell_state_error = np.zeros_like(self.cell_state[0])
        tanh_input_gate_error = np.zeros((self.units, 1))
        sigmoid_input_gate_error = np.zeros((self.units, 1))
        output_gate_error = np.zeros((self.units, 1))
        forget_gate_error = np.zeros((self.units, 1))
        total_weight_error_tanh = np.zeros_like(self.tanh_input_gate.weights)
        total_weight_error_forget = np.zeros_like(self.forget_gate.weights)
        total_weight_error_sigmoid = np.zeros_like(self.sigmoid_input_gate.weights)
        total_weight_error_output = np.zeros_like(self.output_gate.weights)
        total_weight_error_final_output = np.zeros_like(self.final_output_dense.weights)
        total_biases_error_tanh = np.zeros_like(self.tanh_input_gate.biases)
        total_biases_error_forget = np.zeros_like(self.forget_gate.biases)
        total_biases_error_sigmoid = np.zeros_like(self.sigmoid_input_gate.biases)
        total_biases_error_output = np.zeros_like(self.output_gate.biases)
        total_biases_error_final_output = np.zeros_like(self.final_output_dense.biases)
        input_error_array = []
        for i in reversed(range(self.sequence_length + self.output_sequence - 1)):
            if self.final_activation_name == "softmax":
                output_error[i] = np.dot(self.final_activation(self.output[i - self.output_sequence], derivative = True, **self.parameters), output_error)
            else:
                output_error[i] *= self.final_activation(self.unactivated_output[i], True, **self.parameters)
            self.final_output_dense.input = self.hidden_state[i]
            i_e_fo, w_e_fo, b_e_fo = self.final_output_dense.get_errors(output_error[i], optimizer=optimizer, parameters=parameters)
            total_biases_error_final_output += b_e_fo
            total_weight_error_final_output += w_e_fo
            if self.return_sequence:
                i_e_fo = error[i]
            hidden_state_error += i_e_fo
            cell_state_error += tanh(self.cell_state[i], derivative=True) * (hidden_state_error * self.output_gate_output_activated[i])
            output_gate_error = sigmoid(self.output_gate_output[i], derivative=True) * (hidden_state_error * self.output_gate_output_activated[i])
            tanh_input_gate_error = tanh(self.tanh_input_gate_output[i], derivative=True) * (cell_state_error * self.sigmoid_input_gate_output_activated[i])
            sigmoid_input_gate_error = sigmoid(self.sigmoid_input_gate_output[i], derivative=True) * (cell_state_error * self.tanh_input_gate_output_activated[i])
            if i > 0:
                concatenated_input = np.concatenate((self.input[i], self.hidden_state[i - 1]))
                forget_gate_error = sigmoid(self.forget_gate_output[i], derivative=True) * (cell_state_error * self.cell_state[i - 1])
            else:
                concatenated_input = np.concatenate((self.input[i], np.zeros_like(self.hidden_state[i])))
                forget_gate_error = np.zeros_like(forget_gate_error)
            cell_state_error *= self.forget_gate_output_activated[i]
            self.forget_gate.input = concatenated_input
            self.tanh_input_gate.input = concatenated_input
            self.sigmoid_input_gate.input = concatenated_input
            self.output_gate.input = concatenated_input
            i_e_o, w_e_o, b_e_o = self.output_gate.get_errors(output_gate_error, optimizer=optimizer, parameters=parameters)
            total_biases_error_output += b_e_o
            total_weight_error_output += w_e_o
            i_e_s, w_e_s, b_e_s = self.sigmoid_input_gate.get_errors(sigmoid_input_gate_error, optimizer=optimizer, parameters=parameters)
            total_biases_error_sigmoid += b_e_s
            total_weight_error_sigmoid += w_e_s
            i_e_t, w_e_t, b_e_t = self.tanh_input_gate.get_errors(tanh_input_gate_error, optimizer=optimizer, parameters=parameters)
            total_biases_error_tanh += b_e_t
            total_weight_error_tanh += w_e_t
            i_e_f, w_e_f, b_e_f = self.forget_gate.get_errors(forget_gate_error, optimizer=optimizer, parameters=parameters)
            total_biases_error_forget += b_e_f
            total_weight_error_forget += w_e_f
            i_e = i_e_f + i_e_o + i_e_s + i_e_t
            hidden_state_error = i_e[self.x_shape[0]:]
            input_error = i_e[:self.x_shape[0]]
            input_error_array.append(np.copy(input_error))
        total_biases_error_final_output = clip_error(total_biases_error_final_output, limit)
        total_biases_error_forget = clip_error(total_biases_error_forget, limit)
        total_biases_error_output = clip_error(total_biases_error_output, limit)
        total_biases_error_sigmoid = clip_error(total_biases_error_sigmoid, limit)
        total_biases_error_tanh = clip_error(total_biases_error_tanh, limit)
        total_weight_error_final_output = clip_error(total_weight_error_final_output, limit)
        total_weight_error_forget = clip_error(total_weight_error_forget, limit)
        total_weight_error_output = clip_error(total_weight_error_output, limit)
        total_weight_error_sigmoid = clip_error(total_weight_error_sigmoid, limit)
        total_weight_error_tanh = clip_error(total_weight_error_tanh, limit)
        self.final_output_dense.update_learnable_parameters(total_weight_error_final_output, total_biases_error_final_output, learning_rate)
        self.tanh_input_gate.update_learnable_parameters(total_weight_error_tanh, total_biases_error_tanh, learning_rate)
        self.sigmoid_input_gate.update_learnable_parameters(total_weight_error_sigmoid, total_biases_error_sigmoid, learning_rate)
        self.forget_gate.update_learnable_parameters(total_weight_error_forget, total_biases_error_forget, learning_rate)
        self.output_gate.update_learnable_parameters(total_weight_error_output, total_biases_error_output, learning_rate)
        # self.output_biases -= learning_rate * error
        # self.output_weights -= learning_rate * np.dot(error, self.hidden_state[-1].T)
        # error = np.dot(self.output_weights.T, error)
        # cell_state_error = np.zeros_like(self.cell_state[0])
        # tanh_input_gate_error = np.zeros((self.units, 1))
        # sigmoid_input_gate_error = np.zeros((self.units, 1))
        # output_gate_error = np.zeros((self.units, 1))
        # forget_gate_error = np.zeros((self.units, 1))
        # for i in reversed(range(self.sequence_length + self.output_sequence)):
        #     cell_state_error = tanh(self.cell_state[i], derivative=True) * (error * self.output_gate_output_activated[i])
        #     output_gate_error = sigmoid(self.output_gate_output[i], derivative=True) * (error * self.output_gate_output_activated[i])
        #     tanh_input_gate_error = tanh(self.tanh_input_gate_output[i], derivative=True) * (cell_state_error * self.sigmoid_input_gate_output_activated[i])
        #     sigmoid_input_gate_error = sigmoid(self.sigmoid_input_gate_output[i], derivative=True) * (cell_state_error * self.tanh_input_gate_output_activated[i])
        #     if i > 0:
        #         concatenated_input = np.concatenate((self.input[i], self.hidden_state[i - 1]))
        #         forget_gate_error = sigmoid(self.forget_gate_output[i], derivative=True) * (cell_state_error * self.cell_state[i - 1])
        #     else:
        #         concatenated_input = np.concatenate((self.input[i], np.zeros_like(self.hidden_state[i])))
        #         forget_gate_error = np.zeros_like(forget_gate_error)
        #     self.forget_gate.input = concatenated_input
        #     self.tanh_input_gate.input = concatenated_input
        #     self.sigmoid_input_gate.input = concatenated_input
        #     self.output_gate.input = concatenated_input
        #     input_error = self.output_gate.backward(output_gate_error, learning_rate, optimizer, parameters) + self.sigmoid_input_gate.backward(sigmoid_input_gate_error, learning_rate, optimizer, parameters) + self.tanh_input_gate.backward(tanh_input_gate_error, learning_rate, optimizer, parameters) + self.forget_gate.backward(forget_gate_error, learning_rate, optimizer, parameters)
        #     i_error = input_error[:self.x_shape[0]]
        #     error = input_error[self.x_shape[0]:]
        #     input_error = i_error
        if previous_layer_return_sequence:
            return np.array(input_error_array)
        return input_error
    
    def save(self):
        return f"LSTM\n{str(self.units)}\n{str(self.input_shape)}\n{str(self.output_sequence)}\n{str(self.final_activation_name)}\n{str(self.parameters)}\n{str(self.return_sequence)}\n{self.final_output_dense.save()}\n{self.forget_gate.save()}\n{self.tanh_input_gate.save()}\n{self.sigmoid_input_gate.save()}\n{self.output_gate.save()}\n"
    
    def load(self, **data):
        self.return_sequence = data['return_sequence']
        self.final_output_dense = data['final_output_dense']
        self.forget_gate = data['forget_gate']
        self.tanh_input_gate = data['tanh_input_gate']
        self.sigmoid_input_gate = data['sigmoid_input_gate']
        self.output_gate = data['output_gate']
    
    def reset_optimizer(self):
        self.forget_gate.reset_optimizer()
        self.tanh_input_gate.reset_optimizer()
        self.sigmoid_input_gate.reset_optimizer()
        self.output_gate.reset_optimizer()
        self.final_output_dense.reset_optimizer()

class Dropout(BaseLayer):
    def __init__(self, keep_rate):
        self.beta = None
        self.epsilon = None
        self.keep_rate = keep_rate
        self.cores = 1
    
    def compile(self, input_shape):
        return input_shape
    
    def forward(self, input):
        self.input = input
        self.keep = np.greater_equal(np.random.rand(*input.shape), self.keep_rate)
        self.output = self.keep * self.input / self.keep_rate
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        return self.keep * error / self.keep_rate
    
    def save(self):
        return f"Dropout\n{str(self.keep_rate)}\n"
    
    def load(self, **data):
        pass
    
    def reset_optimizer(self):
        pass
    
class LSTM2(BaseLayer):
    def __init__(self, units, input_shape, return_sequence = False):
        self.units = units
        self.input_sequence_length, *self.x_shape = input_shape
        self.x_shape = tuple(self.x_shape)
        self.input_shape = input_shape
        self.return_sequence = return_sequence
        self.forget_gate_layer = Dense(units, (self.x_shape[0] + units, 1), activation="linear")
        self.input_gate_tanh_layer = Dense(units, (self.x_shape[0] + units, 1), activation="linear")
        self.input_gate_sigmoid_layer = Dense(units, (self.x_shape[0] + units, 1), activation="linear")
        self.output_gate_layer = Dense(units, (self.x_shape[0] + units, 1), activation="linear")
        self.hidden_states = np.zeros((self.input_sequence_length, self.units, 1))
        self.cell_states = np.zeros((self.input_sequence_length, self.units, 1))
        self.forget_gate_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.forget_gate_activated_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.input_gate_tanh_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.input_gate_tanh_activated_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.input_gate_sigmoid_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.input_gate_sigmoid_activated_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.output_gate_output = np.zeros((self.input_sequence_length, self.units, 1))
        self.output_gate_activated_output = np.zeros((self.input_sequence_length, self.units, 1))
    
    def compile(self, input_shape):
        if self.input_shape != input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        if self.return_sequence:
            return self.hidden_states.shape
        return self.hidden_states[0].shape
    
    def forward(self, input):
        self.input = input
        self.hidden_states = np.zeros((self.input_sequence_length, self.units, 1))
        self.cell_states = np.zeros((self.input_sequence_length, self.units, 1))
        self.activated_cell_states = np.zeros((self.input_sequence_length, self.units, 1))
        for i in range(self.input_sequence_length):
            concatenated_input = np.concatenate((self.input[i],self.hidden_states[i - 1]))
            self.forget_gate_output[i] = self.forget_gate_layer.forward(concatenated_input)
            self.input_gate_tanh_output[i] = self.input_gate_tanh_layer.forward(concatenated_input)
            self.input_gate_sigmoid_output[i] = self.input_gate_sigmoid_layer.forward(concatenated_input)
            self.output_gate_output[i] = self.output_gate_layer.forward(concatenated_input)
            self.forget_gate_activated_output[i] = sigmoid(self.forget_gate_output[i], derivative=False)
            self.input_gate_tanh_activated_output[i] = tanh(self.input_gate_tanh_output[i], derivative=False)
            self.input_gate_sigmoid_activated_output[i] = sigmoid(self.input_gate_sigmoid_output[i], derivative=False)
            self.output_gate_activated_output[i] = sigmoid(self.output_gate_output[i], derivative=False)
            self.cell_states[i] = (self.cell_states[i-1] * self.forget_gate_activated_output[i]) + (self.input_gate_tanh_activated_output[i] * self.input_gate_sigmoid_activated_output[i])
            self.activated_cell_states[i] = tanh(self.cell_states[i], derivative=False)
            self.hidden_states[i] = self.activated_cell_states[i] * self.output_gate_activated_output[i]
        if self.return_sequence:
            return self.hidden_states
        return self.hidden_states[-1]
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        cell_state_error = np.zeros((self.units, 1))
        input_error = np.zeros(self.input_shape)
        hidden_state_error = np.zeros((self.units, 1))
        t_w_e_si = np.zeros((self.units, self.x_shape[0] + self.units))
        t_w_e_ti = np.zeros((self.units, self.x_shape[0] + self.units))
        t_w_e_o = np.zeros((self.units, self.x_shape[0] + self.units))
        t_w_e_f = np.zeros((self.units, self.x_shape[0] + self.units))
        t_b_e_si = np.zeros((self.units, 1))
        t_b_e_ti = np.zeros((self.units, 1))
        t_b_e_o = np.zeros((self.units, 1))
        t_b_e_f = np.zeros((self.units, 1))
        if not self.return_sequence:
            hidden_state_error = error
        for i in reversed(range(self.input_sequence_length)):
            if self.return_sequence:
                hidden_state_error += error[i]
            output_gate_error = hidden_state_error * self.activated_cell_states[i] * sigmoid(self.output_gate_output[i], derivative=True)
            cell_state_error += hidden_state_error * self.output_gate_activated_output[i] * tanh(self.cell_states[i], derivative=True)
            input_tanh_error = cell_state_error * self.input_gate_sigmoid_activated_output[i] * tanh(self.input_gate_tanh_output[i], derivative=True)
            input_sigmoid_error = cell_state_error * self.input_gate_tanh_activated_output[i] * sigmoid(self.input_gate_sigmoid_output[i], derivative=True)
            if i > 0:
                forget_gate_error = cell_state_error * self.cell_states[i - 1] * sigmoid(self.forget_gate_output[i], derivative=True)
                concatenated_input = np.concatenate((self.input[i],self.hidden_states[i - 1]))
            else:
                forget_gate_error = np.zeros_like(cell_state_error)
                concatenated_input = np.concatenate((self.input[i],np.zeros_like(self.hidden_states[0])))
            self.forget_gate_layer.input = concatenated_input
            self.input_gate_sigmoid_layer.input = concatenated_input
            self.input_gate_tanh_layer.input = concatenated_input
            self.output_gate_layer.input = concatenated_input
            i_e_o, w_e_o, b_e_o = self.output_gate_layer.get_errors(output_gate_error, optimizer=optimizer, parameters=parameters)
            i_e_si, w_e_si, b_e_si = self.input_gate_sigmoid_layer.get_errors(input_sigmoid_error, optimizer=optimizer, parameters=parameters)
            i_e_ti, w_e_ti, b_e_ti = self.input_gate_tanh_layer.get_errors(input_tanh_error, optimizer=optimizer, parameters=parameters)
            i_e_f, w_e_f, b_e_f = self.forget_gate_layer.get_errors(forget_gate_error, optimizer=optimizer, parameters=parameters)
            concatenated_input_error = i_e_f + i_e_ti + i_e_si + i_e_o
            t_w_e_o += w_e_o
            t_w_e_si += w_e_si
            t_w_e_ti += w_e_ti
            t_w_e_f += w_e_f
            t_b_e_o += b_e_o
            t_b_e_si += b_e_si
            t_b_e_ti += b_e_ti
            t_b_e_f += b_e_f
            input_error[i] = np.copy(concatenated_input_error[:self.x_shape[0]])
            hidden_state_error = np.copy(concatenated_input_error[self.x_shape[0]:])
            cell_state_error *= self.forget_gate_activated_output[i]
        self.output_gate_layer.update_learnable_parameters(t_w_e_o, t_b_e_o, learning_rate)
        self.input_gate_tanh_layer.update_learnable_parameters(t_w_e_ti, t_b_e_ti, learning_rate)
        self.input_gate_sigmoid_layer.update_learnable_parameters(t_w_e_si, t_b_e_si, learning_rate)
        self.forget_gate_layer.update_learnable_parameters(t_w_e_f, t_b_e_f, learning_rate)
        return input_error
    
    def save(self):
        return f"LSTM2\n{str(self.units)}\n{str(self.input_shape)}\n{str(self.return_sequence)}\n{self.forget_gate_layer.save()}\n{self.input_gate_tanh_layer.save()}\n{self.input_gate_sigmoid_layer.save()}\n{self.output_gate_layer.save()}\n"
    
    def load(self, **data):
        self.forget_gate_layer = data['forget_gate']
        self.input_gate_tanh_layer = data['tanh_input_gate']
        self.input_gate_sigmoid_layer = data['sigmoid_input_gate']
        self.output_gate_layer = data['output_gate']
    
    def reset_optimizer(self):
        self.forget_gate_layer.reset_optimizer()
        self.input_gate_tanh_layer.reset_optimizer()
        self.input_gate_sigmoid_layer.reset_optimizer()
        self.output_gate_layer.reset_optimizer()

class RepeatVector(BaseLayer):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
    
    def compile(self, input_shape):
        self.output_shape = (self.sequence_length, *input_shape)
        return (self.sequence_length, *input_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.repeat(input[np.newaxis,:], self.sequence_length, 0)
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        return np.sum(error,0)
    
    def save(self):
        return f"RepeatVector\n{self.sequence_length}\n"
    
    def load(self, **data):
        pass
    
    def reset_optimizer(self):
        pass
    
class TimeDistributed(BaseLayer):
    def __init__(self, layer):
        if not isinstance(layer,Dense):
            raise Exception("TimeDistributed only takes layers that are Dense.")
        self.layer = layer
    
    def compile(self, input_shape):
        self.input_shape = input_shape
        self.sequence_length, *self.layer_input_shape = input_shape
        self.layer_input_shape = tuple(self.layer_input_shape)
        self.layer_output_shape = self.layer.compile(self.layer_input_shape)
        self.output_shape = (self.sequence_length, *self.layer_output_shape)
        return self.output_shape
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros((self.output_shape))
        for i in range(self.sequence_length):
            self.output[i] = self.layer.forward(self.input[i])
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        t_i_e = []
        i_e, t_w_e, t_b_e = self.layer.get_errors(error[-1], optimizer=optimizer, parameters = parameters)
        t_i_e.insert(0,i_e)
        for i in reversed(range(self.sequence_length - 1)):
            self.layer.forward(self.input[i])
            i_e, w_e, b_e = self.layer.get_errors(error[i], optimizer=optimizer, parameters = parameters)
            t_w_e += w_e
            t_b_e += b_e
            t_i_e.insert(0,i_e)
        self.layer.update_learnable_parameters(t_w_e, t_b_e, learning_rate)
        t_i_e = np.array(t_i_e).reshape((len(t_i_e), *i_e.shape))
        return t_i_e
    
    def save(self):
        return f"TimeDistributed\n{str(self.input_shape)}\n{str(self.output_shape)}\n{self.layer.save()}\n"
    
    def load(self, **data):
        self.input_shape = data['input_shape']
        self.sequence_length, *self.layer_input_shape = data['input_shape']
        self.layer_input_shape = tuple(self.layer_input_shape)
        self.output_shape = data['output_shape']
    
    def reset_optimizer(self):
        self.layer.reset_optimizer()

class Attention(BaseLayer):
    def __init__(self, heads, input_shape, vector_size, output_size):
        self.heads = heads
        self.vector_size = vector_size
        self.output_size = output_size
        sequence_length,input_size = input_shape
        self.input_shape = input_shape
        self.query_weights = np.random.uniform(-1./np.sqrt(input_size),1./np.sqrt(input_size),(heads, input_size, vector_size))
        self.value_weights = np.random.uniform(-1./np.sqrt(input_size),1./np.sqrt(input_size),(heads, input_size, vector_size))
        self.key_weights = np.random.uniform(-1./np.sqrt(input_size),1./np.sqrt(input_size),(heads, input_size, vector_size))
        self.output_weights = np.random.uniform(-1./np.sqrt(vector_size*heads),1./np.sqrt(vector_size*heads),(vector_size*heads,output_size))
        self.output_shape = (sequence_length,output_size)
        self.query_matrix = np.zeros((heads,sequence_length,vector_size))
        self.value_matrix = np.zeros((heads,sequence_length,vector_size))
        self.key_matrix = np.zeros((heads,sequence_length,vector_size))
        self.score = np.zeros((heads, sequence_length, sequence_length))
        self.attention = np.zeros((heads, sequence_length, sequence_length))
        self.final_value = np.zeros((heads, sequence_length, vector_size))
        self.s_value = np.zeros((self.output_shape))
        self.average_error = np.zeros((self.output_shape))
        self.output_biases = np.zeros((self.output_shape))
    
    def compile(self, input_shape):
        if self.input_shape != input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        return self.output_shape
    
    def forward(self, input):
        self.input = np.reshape(input, self.input_shape)
        for i in range(self.heads):
            self.query_matrix[i] = np.dot(self.input, self.query_weights[i])
            self.value_matrix[i] = np.dot(self.input, self.value_weights[i])
            self.key_matrix[i] = np.dot(self.input, self.key_weights[i])
            self.score[i] = np.dot(self.query_matrix[i], self.key_matrix[i].T)/np.sqrt(self.vector_size)
            temp = np.exp(self.score[i] - np.max(self.score[i],1).reshape((len(self.score[i]),1)))
            self.attention[i] = (temp / np.sum(temp,1)) / self.vector_size
            self.final_value[i] = np.dot(self.attention[i], self.value_matrix[i])
            if i == 0:
                self.output_value = self.final_value[i]
            else:
                self.output_value = np.concatenate((self.output_value, self.final_value[i]),1)
        self.output = np.dot(self.output_value, self.output_weights) + self.output_biases
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        error = np.reshape(error,self.output_shape)
        if optimizer == momentum:
            self.average_error = momentum(self.average_error, error, **parameters)
            error = self.average_error
        elif optimizer == rmsprop:
            self.s_value, error = rmsprop(self.s_value, error, **parameters)
        elif optimizer == adam:
            self.s_value, self.average_error, error = adam(self.s_value, self.average_error, error, **parameters)
        elif optimizer == None or no_optimizer:
            pass
        else:
            raise Exception("Invalid optimizer")
        output_weight_error = np.dot(self.output_value.T, error)
        self.output_biases -= learning_rate * error
        self.output_weights -= learning_rate * output_weight_error
        total_value_error = np.dot(error, self.output_weights.T)
        final_value_errors = np.split(total_value_error, self.heads, 1)
        input_error = np.zeros_like(self.input)
        for i in range(self.heads):
            value_error = np.dot(self.attention[i].T, final_value_errors[i])
            attention_error = np.dot(final_value_errors[i], self.value_matrix[i].T)
            softmax_error = np.zeros_like(attention_error)
            for j in range(len(attention_error)):
                temp = np.tile(attention_error[j].reshape((len(attention_error[j]),1)),len(attention_error[j]))
                softmax_error[j] = np.dot(attention_error[j], (np.identity(len(attention_error[j])) - temp) * temp.T)
            softmax_error /= self.vector_size
            query_error = np.dot(softmax_error, self.key_matrix[i])
            key_error = np.dot(self.query_matrix[i].T, softmax_error).T
            input_error += np.dot(value_error, self.value_weights[i].T) + np.dot(query_error, self.query_weights[i].T) + np.dot(key_error, self.key_weights[i].T)
            self.value_weights[i] -= learning_rate * np.dot(self.input.T, value_error)
            self.key_weights[i] -= learning_rate * np.dot(self.input.T, key_error)
            self.query_weights[i] -= learning_rate * np.dot(self.input.T, query_error)
        return input_error
    
    def save(self):
        return f"Attention\n{str(self.heads)}\n{str(self.input_shape)}\n{str(self.vector_size)}\n{str(self.output_size)}\n{str(denumpy(self.query_weights))}\n{str(denumpy(self.value_weights))}\n{str(denumpy(self.key_weights))}\n{str(denumpy(self.output_weights))}\n{str(denumpy(self.output_biases))}\n"
    
    def load(self, **data):
        self.query_weights = data['query_weights']
        self.key_weights = data['key_weights']
        self.value_weights = data['value_weights']
        self.output_weights = data['output_weights']
        self.output_biases = data['output_biases']
    
    def reset_optimizer(self):
        self.average_error = np.zeros_like(self.average_error)
        self.s_value = np.zeros_like(self.s_value)

class Activation(BaseLayer):
    def __init__(self, activation, parameters = {}):
        self.activation_name = activation
        self.activation = get_function(activation)
        self.parameters = parameters
        verify_activation(self.activation, parameters)
    
    def compile(self, input_shape):
        return input_shape
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input, derivative=False, **self.parameters)
        return self.output

    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        if self.activation_name == "softmax":
            error = np.dot(self.activation(self.output,derivative = True, **self.parameters), error)
        else:
            error *= self.activation(self.input, derivative = True, **self.parameters)
        return error
    
    def save(self):
        return f"Activation\n{str(self.activation_name)}\n{str(self.parameters)}\n"
    
    def load(self, **data):
        pass

    def reset_optimizer(self):
        pass

class Reshape(BaseLayer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def compile(self, input_shape):
        if self.input_shape != input_shape:
            raise Exception(f"Input shape from previous layer {input_shape} does not match declared input shape {self.input_shape}")
        try:
            np.zeros(self.input_shape).reshape(self.output_shape)
        except:
            raise Exception(f"Cannot reshape array of size {np.product(self.input_shape)} to {self.output_shape}")
        return self.output_shape
    
    def forward(self, input):
        self.input = input
        self.output = np.reshape(self.input, self.output_shape)
        return self.output
    
    def backward(self, error, learning_rate, optimizer=None, parameters={}):
        return np.reshape(error, self.input_shape)
    
    def save(self):
        return f"Reshape\n{str(self.input_shape)}\n{str(self.output_shape)}\n"

    def load(self, **data):
        pass
    
    def reset_optimizer(self):
        pass