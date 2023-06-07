# importations
import cmath as cm
import random as rn
import time

# functions to perform matrix operations
# to get a reccursive copy of a matrix
def deep_copy(matrix):
    try:
        len(matrix)
        result = []
        for i in range(len(matrix)):
            result.append(deep_copy(matrix[i]))
        return result
    except:
        return matrix

# to make sure value is a row matrix or a matrix
def matrix(arr, dimensions):
    def fail_at(arr, dim):
        while dim > 0:
            try:
                len(arr)
                dim -= 1
                arr = arr[0]
            except:
                return dim
        return dim
    result = fail_at(arr, dimensions)
    for i in range(result):
        arr = list(arr for j in range(1))
    return arr

# to perform sum of matrices
def matrix_sum(*args, **kwargs):
    if 'args' in kwargs:
        args = kwargs.get('args')
    if isinstance(args[0], list):
        result = []
        for i in range(len(args[0])):
            arr = []
            for j in range(len(args)):
                arr.append(args[j][i])
            result.append(matrix_sum(args = arr))
        return result
    result = 0
    for arg in args:
        result += arg
    return result

def matrix_sum(a, b):
    if isinstance(a, list):
        result = []
        for i in range(len(a)):
            result.append(matrix_sum(a[i], b[i]))
        return result
    return a + b

# to perform matrix difference
def matrix_difference(a, b):
    if isinstance(a, list):
        result = []
        for i in range(len(a)):
            result.append(matrix_difference(a[i], b[i]))
        return result
    return a - b

# to perform scalar product of a matrix
def matrix_scalar_product(scalar, matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(matrix_scalar_product(scalar, matrix[i]))
        return result
    return scalar * matrix

# to perform scalar division of a matrix
def matrix_scalar_division(scalar, matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(matrix_scalar_division(scalar, matrix[i]))
        return result
    return matrix / scalar

# to perform matrix scalar subtraction
def matrix_scalar_difference(scalar, matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(matrix_scalar_difference(scalar, matrix[i]))
        return result
    return matrix - scalar

# to perform proper matrix product
def matrix_product(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            ans = 0
            for k in range(len(b)):
                ans += a[i][k] * b[k][j]
            row.append(ans)
        result.append(row)
    return result

# to perform elementwise product
def elementwise_product(a, b):
    if isinstance(a, list):
        result = []
        for i in range(len(a)):
            result.append(elementwise_product(a[i], b[i]))
        return result
    return a * b

# to perform elementwise product
def elementwise_division(a, b):
    if isinstance(a, list):
        result = []
        for i in range(len(a)):
            result.append(elementwise_division(a[i], b[i]))
        return result
    return a / b

# to perform matrix transpose
def matrix_transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        result.append(row)
    return result

# to get e^element of matrix elements
def e_raise_matrix(matrix):
    try:
        len(matrix)
        result = []
        for i in range(len(matrix)):
            result.append(e_raise_matrix(matrix[i]))
        return result
    except:
        return (cm.e ** matrix).real

def sum_of_elements(matrix):
    try:
        len(matrix)
        result = 0
        for i in range(len(matrix)):
            result += (sum_of_elements(matrix[i]))
        return result
    except:
        return matrix

def max_of_elements(matrix):
    try:
        len(matrix)
        result = []
        for i in range(len(matrix)):
            result.append(max_of_elements(matrix[i]))
        return max(result)
    except:
        return matrix

# activation functions
# hyperbolic tangent function
def tanh(matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(tanh(matrix[i]))
        return result
    val = cm.tanh(matrix).real
    return val

def der_tanh(matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(der_tanh(matrix[i]))
        return result
    val = cm.tanh(matrix).real
    return (1 - val**2)

def sigmoid(matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(sigmoid(matrix[i]))
        return result
    val = cm.e**matrix
    return (1 / (1 + 1 / val))

def der_sigmoid(matrix):
    if isinstance(matrix, list):
        result = []
        for i in range(len(matrix)):
            result.append(der_sigmoid(matrix[i]))
        return result
    val = cm.e**matrix
    return (val / ((val + 1)**2))

def relu(matrix):
    try:
        size = len(matrix)
        result = []
        for i in range(size):
            result.append(relu(matrix[i]))
        return result
    except:
        if matrix > 0:
            return matrix
        return 0

def der_relu(matrix):
    try:
        size = len(matrix)
        result = []
        for i in range(size):
            result.append(relu(matrix[i]))
        return result
    except:
        if matrix > 0:
            return 1
        return 0

# error functions
# mean square error function
def mse(y_true,y_predicted):
    result = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            result += ((y_true[i][j] - y_predicted[i][j])**2)
    result /= len(y_true)
    return result.real

def der_mse(y_true,y_predicted):
    result = list(list(0 for i in range(len(y_true[0]))) for j in range(len(y_true)))
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            result[i][j] += ((y_predicted[i][j] - y_true[i][j])*2)
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] /= len(y_true)
    return result

# binary cross entropy method
def binary_cross_entorpy(y_true, y_pred):
    result =0
    # print(y_pred,y_true)
    for i in range(len(y_true[0])):
        result += y_true[0][i] * cm.log(y_pred[0][i]).real + (1 - y_true[0][i]) * cm.log(1 - y_pred[0][i]).real
    result /= len(y_true[0])
    return -result

def der_binary_cross_entropy(y_true, y_pred):
    result = []
    row = []
    for i in range(len(y_true[0])):
        row.append(((1 - y_true[0][i])/(1 - y_pred[0][i]) - (y_true[0][i] / y_pred[0][i])) / len(y_true[0]))
    result.append(row)
    return result

# categorical cross entopy method
def categorical_cross_entorpy(y_true, y_pred):
    result = 0
    for i in range(len(y_true[0])):
        result += y_true[0][i] * cm.log(y_pred[0][i]).real
    # result /= len(y_true[0])
    return -result

def der_categorical_cross_entropy(y_true, y_pred):
    result = []
    row = []
    for i in range(len(y_true[0])):
        row.append(-(y_true[0][i] / y_pred[0][i]))
    result.append(row)
    return result

# functions for the convolutional layer
# funciton to perform valid correlation
def valid_correlate(matrix, kernel):
    result = []
    kernel_size = len(kernel)
    end_row = len(matrix) - kernel_size + 1
    end_column = len(matrix[0]) - kernel_size + 1
    for i in range(end_row):
        row = []
        for j in range(end_column):
            ans = 0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    ans += matrix[i + k][j + l] * kernel[k][l]
            row.append(ans)
        result.append(row)
    return result

# to perform full correlation
def full_correlate(matrix, kernel):
    kernel_size = len(kernel)
    damping = kernel_size - 1
    new_matrix = []
    for i in range(damping):
        new_matrix.append(list(0 for j in range(len(matrix[0]) + 2 * damping)))
    for i in range(len(matrix)):
        arr = list(0 for j in range(damping))
        arr.extend(matrix[i])
        arr.extend(list(0 for j in range(damping)))
        new_matrix.append(arr)
    for i in range(damping):
        new_matrix.append(list(0 for j in range(len(matrix[0]) + 2 * damping)))
    return valid_correlate(new_matrix, kernel)

# to perform matrix rotation by 180 degrees
def rotate_matrix_180(matrix):
    result = []
    for i in reversed(range(len(matrix))):
        row = []
        for j in reversed(range(len(matrix[i]))):
            row.append(matrix[i][j])
        result.append(row)
    return result

# to perform valid convolution
def valid_convolve(matrix, kernel):
    new_kernel = rotate_matrix_180(kernel)
    return valid_correlate(matrix, new_kernel)

# to perform full convolution
def full_convolve(matrix, kernel):
    new_kernel = rotate_matrix_180(kernel)
    return full_correlate(matrix, new_kernel)

# general layer class
class Layer():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

# dense layer class
class DenseLayer(Layer):
    def __init__(self, inputs, outputs):
        self.bias = list(list((rn.random()) for j in range(outputs)) for i in range(1))
        self.weights = list(list((rn.random()) for j in range(outputs)) for i in range(inputs))
    
    def forward(self, input):
        self.input = input
        self.output = matrix_sum(matrix_product(self.input, self.weights), self.bias)
        return self.output
    
    def backward(self, output_error, learning_rate):
        self.bias = matrix_difference(self.bias, matrix_scalar_product(learning_rate, output_error))
        input_error = matrix_product(output_error, matrix_transpose(self.weights))
        self.weights = matrix_difference(self.weights, matrix_scalar_product(learning_rate, matrix_product(matrix_transpose(self.input), output_error)))
        return input_error

# activation layer class
class ActivationLayer(Layer):
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
    
    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return elementwise_product(self.derivative(self.input), output_error)

# convolutional layer class
class ConvolutionLayer(Layer):
    def __init__(self, input_shape, kernel_size, output_depth):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.output_shape = (output_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (output_depth, input_depth, kernel_size, kernel_size)
        self.kernels = list(list(list(list((rn.random()) for k in range(kernel_size)) for j in range(kernel_size)) for i in range(input_depth)) for h in range(output_depth))
        self.biases = list(list(list((rn.random()) for k in range(self.output_shape[2])) for j in range(self.output_shape[1])) for i in range(output_depth))
    
    def forward(self, input):
        self.input = input
        output = deep_copy(self.biases)
        for i in range(self.kernel_shape[0]):
            for j in range(self.kernel_shape[1]):
                output[i] = matrix_sum(output[i], valid_correlate(input[j], self.kernels[i][j]))
        self.output = output
        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = list(list(list(0 for k in range(self.input_shape[2])) for j in range(self.input_shape[1])) for i in range(self.input_shape[0]))
        kernel_error = list(list(list(list(0 for l in range(self.kernel_shape[3])) for k in range(self.kernel_shape[2])) for j in range(self.kernel_shape[1])) for i in range(self.kernel_shape[0]))
        for j in range(self.input_shape[0]):
            for i in range(self.output_shape[0]):
                input_error[j] = matrix_sum(input_error[j], full_convolve(output_error[i], self.kernels[i][j]))
                kernel_error[i][j] = valid_correlate(self.input[j], output_error[i])
        self.biases = matrix_difference(self.biases, matrix_scalar_product(learning_rate, output_error))
        self.kernels = matrix_difference(self.kernels, matrix_scalar_product(learning_rate, kernel_error))
        return input_error

# pooling layer class
class PoolingLayer(Layer):
    def __init__(self,pool_size,pool_type):
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pool_select = None
    
    def pool(self,input):
        pool_select = list(list(list(0 for i in range(len(input[0][0]))) for j in range(len(input[0]))) for k in range(len(input)))
        result = []
        for i in range(len(input)):
            height = int(len(input[i])/self.pool_size)
            matrix = []
            for j in range(height):
                width = int(len(input[i][j])/self.pool_size)
                row = []
                for k in range(width):
                    arr = []
                    sums = 0
                    maxim = -cm.inf
                    previous = [j*self.pool_size, k*self.pool_size]
                    for l in range(self.pool_size):
                        for m in range(self.pool_size):
                            sums += input[i][j*self.pool_size + l][k*self.pool_size + m]
                            if input[i][j*self.pool_size + l][k*self.pool_size + m] > maxim:
                                pool_select[i][previous[0]][previous[1]] = 0
                                maxim = input[i][j*self.pool_size + l][k*self.pool_size + m]
                                previous = [j*self.pool_size + l, k*self.pool_size + m]
                                pool_select[i][previous[0]][previous[1]] = 1
                            arr.append(input[i][j*self.pool_size + l][k*self.pool_size + m])
                    if self.pool_type == 'max':
                        row.append(max(arr))
                    elif self.pool_type == 'average':
                        row.append(sums/(self.pool_size**2))
                    else:
                        raise NotImplementedError
                matrix.append(row)
            result.append(matrix)
        self.pool_select = pool_select
        return result

    def depool(self,input):
        result = list(list(list(0 for i in range(len(input[0][0]) * self.pool_size)) for j in range(len(input[0]) * self.pool_size)) for k in range(len(input)))
        for i in range(len(result)):
            for j in range(len(result[i])):
                for k in range(len(result[i][j])):
                    if self.pool_select[i][j][k] == 1:
                        result[i][j][k] = input[i][int(j / self.pool_size)][int(k / self.pool_size)]/(self.pool_size)
        return result

    def forward(self, input):
        self.input = input
        self.output = self.pool(input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return self.depool(output_error)
    
# class for reshape layer
class ReshapeLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        def get_data_array(arr):
            data = []
            try:
                len(arr)
                try:
                    len(arr[0])
                    for i in range(len(arr)):
                        data.extend(get_data_array(arr[i]))
                except:
                    data.extend(arr)
            except:
                data.append(arr)
            return data

        def group(arr, number):
            j = 0
            part = []
            new_arr = []
            for i in range(len(arr)):
                if j >= number - 1:
                    j = 0
                    part.append(arr[i])
                    new_arr.append(part)
                    part = []   
                else:
                    part.append(arr[i])
                    j += 1
            return new_arr
        
        self.input = input
        data = get_data_array(input)
        for i in reversed(range(len(self.output_shape))):
            data = group(data, self.output_shape[i])
        self.output = data[0]
        return self.output
    
    def backward(self, output_error, learning_rate):
        def get_data_array(arr):
            data = []
            try:
                len(arr)
                try:
                    len(arr[0])
                    for i in range(len(arr)):
                        data.extend(get_data_array(arr[i]))
                except:
                    data.extend(arr)
            except:
                data.append(arr)
            return data

        def group(arr, number):
            j = 0
            part = []
            new_arr = []
            for i in range(len(arr)):
                if j >= number - 1:
                    j = 0
                    part.append(arr[i])
                    new_arr.append(part)
                    part = []
                else:
                    part.append(arr[i])
                    j += 1
            return new_arr
        
        data = get_data_array(output_error)
        for i in reversed(range(len(self.input_shape))):
            data = group(data, self.input_shape[i])
        return data[0]

# softmax layer
class SoftMaxLayer(Layer):
    def forward(self, input):
        self.input = input
        inp = matrix_scalar_difference(max_of_elements(input),input)
        tmp = e_raise_matrix(inp)
        ddt = matrix_scalar_division(sum_of_elements(tmp), tmp)
        self.output = ddt
        return ddt

    def backward(self, output_error, learning_rate):
        size = len(self.output[0])
        tmp = []
        for i in range(size):
            tmp.append(deep_copy(self.output[0]))
        identity = []
        for i in range(size):
            row = []
            for k in range(size):
                if i == k:
                    row.append(1)
                else:
                    row.append(0)
            identity.append(row)
        input_error = matrix_product(output_error, elementwise_product(tmp, matrix_difference(identity, matrix_transpose(tmp))))
        return input_error

# network class
class Network():
    def __init__(self, *args, **kwargs):
        self.layers = []
        self.loss = kwargs.get('loss')
        self.derivative = kwargs.get('loss_derivative')
        for arg in args:
            self.layers.append(arg)
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.derivative = loss_derivative
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        avv = 0
        for i in range(epochs):
            start = time.time()
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                err += self.loss(y_train[j], output)
                output_error = self.derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, learning_rate)
            err /= samples
            print(f"epoch {i+1}/{epochs} \t error: {err}")
            avv += time.time() - start
            print("Time:",time.time() - start,"\tETA:", (avv/(i+1)) * (epochs - (i +1)))
    
    def predict(self, x_test):
        result = []
        for i in range(len(x_test)):
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result