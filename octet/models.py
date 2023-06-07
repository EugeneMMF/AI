from octet.subfunctions import get_function, get_optimizer, get_error_function
import numpy as np
from time import time, strftime, gmtime
from octet.layers import *
import copy
# from layers import *
# from subfunctions import get_function, get_optimizer, get_error_function

class Sequential():
    def __init__(self, *args, descriptor = ''):
        """Creates a sequential model
        Optional arguments of layers of type ennm.layers may be added in order of occurrence in the model
        """
        self.layers = []
        self.descriptor = descriptor
        for arg in args:
            self.layers.append(arg)
    
    def add(self, layer):
        """Adds a layer to the model sequentially

        Args:
            layer (ennm.layer): layer object from ennm.layers.
            May be Conv2D, Dense, Flatten, Pool2D, Input or Dropout.
        """
        self.layers.append(layer)

    def compile(self, error_function, optimizer = None, parameters = {}):
        self.optimizer_name = optimizer
        self.optimizer = get_optimizer(optimizer)
        self.parameters = parameters
        verify_activation(self.optimizer, parameters)
        self.error_function_name = error_function
        self.error_function = get_error_function(self.error_function_name)
        input_shape = self.layers[0].input_shape
        for layer in self.layers:
            input_shape = layer.compile(input_shape)
    
    def fit(self, x_train, y_train, epochs, learning_rate = 0.01, accuracy = False, filename = None, save_after = "each", adaptive_learning_rate = False, batch_size = 0,per_object = False):
        """Fit the model to the data provided in x_train to the output y_train.

        Args:
            x_train (numpy.array or list): a list or numpy array of the training input data sequence
            y_train (numpy.array or list): a list or numpy array of the expected output data sequence
            epochs (int): number of iterations to train the model on the entire dataset
            accuracy (bool): if set to True gives accuracy data of the model on the training data at each epoch
            learning_rate (float): number between 0 and 1 that serves as the learning rate of the model during training (defaults to 0.01)
            filename (str): if provided, saves the trained model to the file for later use
            save_after(int or 'all' or 'each'): if 'all' saves to the filename after all epochs of training are complete, if 'each' saves to the filename after each epoch, if int n saves after every n epochs
            adaptive_learning_rate (float): float of the percentage decrease in the learnig rate when training when the model increases the error.
        """
        if batch_size == 0:
            batch_size = len(x_train)
        previous_error = 0
        optimizer = self.optimizer
        total_time = 0
        for i in range(epochs):
            print_statement = f"epoch {i + 1}/{epochs}"
            start = time()
            error = 0
            indices = list(range(len(x_train)))
            np.random.shuffle(indices)
            x_train_batch,y_train_batch = x_train[indices][:batch_size],y_train[indices][:batch_size]
            dt = 0
            for ind,(x,y) in enumerate(zip(x_train_batch, y_train_batch)):
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                e = self.error_function(y, output)
                if per_object:
                    print(f"Error in case {ind}: {e}")
                error += e
                output_error = self.error_function(y, output, derivative = True)
                for layer in reversed(range(len(self.layers))):
                    if isinstance(self.layers[layer], LSTM) and isinstance(self.layers[layer - 1],LSTM):
                        output_error = self.layers[layer].backward(output_error, learning_rate, optimizer = optimizer, parameters = self.parameters, previous_layer_return_sequence=True)
                    else:
                        output_error = self.layers[layer].backward(output_error, learning_rate, optimizer = optimizer, parameters = self.parameters)
                dt += 1
            # for layer in self.layers:
            #     layer.reset_optimizer()
            error /= len(x_train_batch)
            print_statement += f"\terror = {error}"
            if accuracy:
                count = 0
                for x,y in zip(x_train, y_train):
                    output = x
                    for layer in self.layers:
                        if isinstance(layer, Dropout):
                            continue
                        output = layer.forward(output)
                    if np.all(np.round(np.equal(output, np.max(output)) * 1) == y):
                        count += 1
                ratio = count / len(x_train)
                end = time()
                tm = end - start
                total_time += tm
                # print(f"epoch {i + 1}/{epochs}\terror = {error}\ttime: {strftime('%H hrs %M min %S sec', gmtime(tm))}\tETA: {strftime('%H hrs %M min %S sec', gmtime(total_time / (i + 1) * (epochs - i - 1)))}\taccuracy = {ratio}")
            else:
                end = time()
                tm = end - start
                total_time += tm
                # print(f"epoch {i + 1}/{epochs}\terror = {error}\ttime: {strftime('%H hrs %M min %S sec', gmtime(tm))}\tETA: {strftime('%H hrs %M min %S sec', gmtime(total_time / (i + 1) * (epochs - i - 1)))}")
            print_statement += f"\ttime: {strftime('%H hrs %M min %S sec', gmtime(tm))}\tETA: {strftime('%H hrs %M min %S sec', gmtime(total_time / (i + 1) * (epochs - i - 1)))}"
            if accuracy:
                print_statement += f"\taccuracy = {ratio}"
            if isinstance(adaptive_learning_rate, float) and previous_error != 0:
                if error > previous_error:
                    learning_rate *= (1 - adaptive_learning_rate)
                    print_statement += f"\tnew learning_rate = {learning_rate}"
                # print("New learning rate:", learning_rate)
            previous_error = error
            print(print_statement)
            if filename:
                if save_after == "each":
                    self.save(filename)
                elif isinstance(save_after, int) and ((i + 1) % save_after) == 0:
                    self.save(filename)
        if filename and (save_after == "all" or (isinstance(save_after, int) and ((i+1) % save_after) != 0)):
            self.save(filename)
    
    def predict(self, x_test, accuracy = False):
        """Gives a list of predictions of the model to the input list given

        Args:
            x_test (numpy.array or list): a list or numpy array containing all the inputs for which the predictions are to be made
            accuracy (bool): if set to True, y_test must also be provided for which the an accuracy score of the predictions shall be returned alongside the predictions
            y_test (numpy.array or list): a list or numpy array containing all the outputs of the inputs provided. Is only needed if accuracy = True

        Returns:
            numpy.array: a numpy array of the predictions of the model to the input list
            numpy.array, float: returns a float for the accuracy of the model in the prediction of the outputs based on the inputs
        """
        results = []
        accuracy = False
        if accuracy:
            pass
        else:
            y_test = x_test
        count = 0
        for x,y in zip(x_test, y_test):
            output = x
            for layer in self.layers:
                if isinstance(layer, Dropout):
                    continue
                output = layer.forward(output)
            if accuracy:
                if np.all(np.round(np.equal(output, np.max(output)) * output) == y):
                    count += 1
            results.append(output)
        if accuracy:
            return np.array(results), count / len(x_test)
        return np.array(results)
    
    def save(self, filename):
        """Saves the model to a file provided

        Args:
            filename (str): the name of the file to which the model is to be saved
        """
        with open(filename, 'w') as f:
            f.write(f"{self.descriptor}\nSequential\n{str(self.error_function_name)}\n{str(self.optimizer_name)}\n{str(self.parameters)}\n")
            for layer in self.layers:
                f.write(layer.save())
            f.write("END")
        f.close
        print(f"Model saved successfully to {filename}")
        
    def load(self, filename):
        """Loads a model from a saved model file

        Args:
            filename (str): the name of the file from which the model is to be loaded
        """
        def get_data(mystr, data_type, list_type):
            mystr = mystr.replace('(','').replace(')','').replace('[','').replace(']','').replace(' ','')
            result = list_type(map(data_type, mystr.split(',')))
            return result

        def group(arr, number):
            result = []
            length = len(arr)
            grp = []
            j = 0
            for i in range(length):
                if j == number - 1:
                    grp.append(arr[i])
                    j = 0
                    result.append(grp)
                    grp = []
                else:
                    grp.append(arr[i])
                    j += 1
            return result
        
        self.layers = []
        with open(filename, "r") as w:
            line = 0
            f = w.readlines()
            total_lines = len(f)
            while line < total_lines:
                if line == 0:
                    while f[line].replace('\n','') != "Sequential":
                        self.descriptor += str(f[line])
                        line += 1
                    self.descriptor = self.descriptor[:-1]
                if f[line].replace('\n','') == "Sequential":
                    line += 1
                    self.error_function_name = f[line].replace('\n','')
                    self.error_function = get_error_function(self.error_function_name)
                    line += 1
                    self.optimizer_name = f[line].replace('\n','')
                    self.optimizer = get_optimizer(self.optimizer_name)
                    line += 1
                    self.parameters = eval(f[line].replace('\n',''))
                    line += 1
                elif f[line].replace('\n','') == "Input":
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    activation = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    layer = Input(input_shape)
                    layer.load(activation = activation, parameters = parameters)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Dense":
                    line += 1
                    nodes = int(f[line].replace('\n',''))
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    output_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    weights_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    activation = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    line += 1
                    weights = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    biases = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    biases = np.array(biases).reshape(output_shape)
                    weights = np.array(weights).reshape(weights_shape)
                    layer = Dense(nodes)
                    layer.load(input_shape = input_shape, output_shape = output_shape, activation = activation, parameters = parameters, weights = weights, biases = biases)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Conv2D":
                    line += 1
                    filters = int(f[line].replace('\n',''))
                    line += 1
                    filter_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    kernel_size = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    output_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    kernels_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    padding = f[line].replace('\n','')
                    line += 1
                    strides = int(f[line].replace('\n',''))
                    line += 1
                    activation = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    line += 1
                    kernels = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    biases = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    kernels = np.array(kernels).reshape(kernels_shape)
                    biases = np.array(biases).reshape(output_shape)
                    layer = Conv2D(filters, kernel_size, input_shape, strides, padding, activation, parameters)
                    layer.compile(input_shape)
                    layer.load(filters = filters, filter_shape = filter_shape, input_shape = input_shape, output_shape = output_shape, padding = padding, strides = strides, activation = activation, kernels = kernels, biases = biases, parameters = parameters)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Flatten":
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    output_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    layer = Flatten()
                    layer.load(input_shape = input_shape, output_shape = output_shape)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "MaxPool2D":
                    line += 1
                    pool_size = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    output_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    layer = MaxPool2D(pool_size)
                    layer.load(input_shape = input_shape, output_shape = output_shape)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "AveragePool2D":
                    line += 1
                    pool_size = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    output_shape = get_data(f[line].replace('\n',''), int, tuple)
                    line += 1
                    layer = AveragePool2D(pool_size)
                    layer.load(input_shape = input_shape, output_shape = output_shape)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "SimpleRNN":
                    line += 1
                    units = int(f[line].replace('\n',''))
                    line += 1
                    input_shape = get_data(f[line].replace('\n',''), int, tuple)
                    seqence_length, *x_shape = input_shape
                    line += 1
                    activation = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    line += 1
                    hidden_biases = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    hidden_weights = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    input_weights = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    output_weights = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    output_biases = get_data(f[line].replace('\n',''), float, list)
                    line += 1
                    hidden_biases = np.array(hidden_biases).reshape((units, 1))
                    hidden_weights = np.array(hidden_weights).reshape((units, units))
                    input_weights = np.array(input_weights).reshape((units, x_shape[0]))
                    output_weights = np.array(output_weights).reshape((x_shape[0], units))
                    output_biases = np.array(output_biases).reshape(x_shape)
                    layer = SimpleRNN(units, input_shape, activation, parameters)
                    layer.compile(input_shape)
                    layer.load(paramters = parameters, hidden_biases = hidden_biases, hidden_weights = hidden_weights, input_weights = input_weights, output_biases = output_biases, output_weights = output_weights)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "LSTM":
                    line += 1
                    units = int(f[line].replace('\n',''))
                    line += 1
                    input_main_shape = get_data(f[line].replace('\n',''), int, tuple)
                    _, *x_main_shape = input_main_shape
                    line += 1
                    output_sequence = int(f[line].replace('\n',''))
                    line += 1
                    final_activation_name = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    line += 1
                    return_sequence = eval(f[line].replace('\n',''))
                    line += 1
                    values = {
                        'final_output_dense': '',
                        'forget_gate': '',
                        'tanh_input_gate': '',
                        'sigmoid_input_gate': '',
                        'output_gate': ''
                    }
                    keys = list(values.keys())
                    for i in range(len(keys)):
                        line += 1
                        nodes = int(f[line].replace('\n',''))
                        line += 1
                        input_shape = get_data(f[line].replace('\n',''), int, tuple)
                        _, *x_shape = input_shape
                        line += 1
                        output_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        weights_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        activation = f[line].replace('\n','')
                        line += 1
                        parameters = eval(f[line].replace('\n',''))
                        line += 1
                        weights = get_data(f[line].replace('\n',''), float, list)
                        line += 1
                        biases = get_data(f[line].replace('\n',''), float, list)
                        line += 2
                        biases = np.array(biases).reshape(output_shape)
                        weights = np.array(weights).reshape(weights_shape)
                        layer = Dense(nodes)
                        layer.load(input_shape = input_shape, output_shape = output_shape, activation = activation, parameters = parameters, weights = weights, biases = biases)
                        values[keys[i]] = layer
                    # line += 1
                    layer = LSTM(units, input_main_shape, output_sequence, final_activation_name, parameters)
                    layer.load(return_sequence = return_sequence, **values)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "LSTM2":
                    line += 1
                    units = int(f[line].replace('\n',''))
                    line += 1
                    input_main_shape = get_data(f[line].replace('\n',''), int, tuple)
                    _, *x_main_shape = input_main_shape
                    line += 1
                    return_sequence = eval(f[line].replace('\n',''))
                    line += 1
                    values = {
                        'forget_gate': '',
                        'tanh_input_gate': '',
                        'sigmoid_input_gate': '',
                        'output_gate': ''
                    }
                    keys = list(values.keys())
                    for i in range(len(keys)):
                        line += 1
                        nodes = int(f[line].replace('\n',''))
                        line += 1
                        input_shape = get_data(f[line].replace('\n',''), int, tuple)
                        _, *x_shape = input_shape
                        line += 1
                        output_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        weights_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        activation = f[line].replace('\n','')
                        line += 1
                        parameters = eval(f[line].replace('\n',''))
                        line += 1
                        weights = get_data(f[line].replace('\n',''), float, list)
                        line += 1
                        biases = get_data(f[line].replace('\n',''), float, list)
                        line += 2
                        biases = np.array(biases).reshape(output_shape)
                        weights = np.array(weights).reshape(weights_shape)
                        layer = Dense(nodes)
                        layer.load(input_shape = input_shape, output_shape = output_shape, activation = activation, parameters = parameters, weights = weights, biases = biases)
                        values[keys[i]] = layer
                    # line += 1
                    layer = LSTM2(units, input_main_shape, return_sequence = return_sequence)
                    layer.load(**values)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "RepeatVector":
                    line += 1
                    sequence_length = eval(f[line].replace('\n',''))
                    line += 1
                    layer = RepeatVector(sequence_length)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "TimeDistributed":
                    line += 1
                    input_main_shape = eval(f[line].replace('\n',''))
                    line += 1
                    output_main_shape = eval(f[line].replace('\n',''))
                    line += 1
                    for i in range(1):
                        line += 1
                        nodes = int(f[line].replace('\n',''))
                        line += 1
                        input_shape = get_data(f[line].replace('\n',''), int, tuple)
                        _, *x_shape = input_shape
                        line += 1
                        output_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        weights_shape = get_data(f[line].replace('\n',''), int, tuple)
                        line += 1
                        activation = f[line].replace('\n','')
                        line += 1
                        parameters = eval(f[line].replace('\n',''))
                        line += 1
                        weights = get_data(f[line].replace('\n',''), float, list)
                        line += 1
                        biases = get_data(f[line].replace('\n',''), float, list)
                        line += 2
                        biases = np.array(biases).reshape(output_shape)
                        weights = np.array(weights).reshape(weights_shape)
                        layer = Dense(nodes)
                        layer.load(input_shape = input_shape, output_shape = output_shape, activation = activation, parameters = parameters, weights = weights, biases = biases)
                    layer = TimeDistributed(layer)
                    layer.load(input_shape = input_main_shape, output_shape = output_main_shape)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Attention":
                    line += 1
                    heads = eval(f[line].replace('\n',''))
                    line += 1
                    input_shape = eval(f[line].replace('\n',''))
                    line += 1
                    vector_size = eval(f[line].replace('\n',''))
                    line += 1
                    output_size = eval(f[line].replace('\n',''))
                    line += 1
                    query_weights = eval(f[line].replace('\n',''))
                    line += 1
                    value_weights = eval(f[line].replace('\n',''))
                    line += 1
                    key_weights = eval(f[line].replace('\n',''))
                    line += 1
                    output_weights = eval(f[line].replace('\n',''))
                    line += 1
                    output_biases = eval(f[line].replace('\n',''))
                    line += 1
                    query_weights = np.array(query_weights)
                    value_weights = np.array(value_weights)
                    key_weights = np.array(key_weights)
                    output_biases = np.array(output_biases)
                    output_weights = np.array(output_weights)
                    layer = Attention(heads,input_shape,vector_size,output_size)
                    data = {
                        'query_weights': query_weights,
                        'value_weights': value_weights,
                        'key_weights': key_weights,
                        'output_weights': output_weights,
                        'output_biases': output_biases
                    }
                    layer.load(**data)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Dropout":
                    line += 1
                    keep_rate = float(f[line].replace('\n',''))
                    line += 1
                    layer = Dropout(keep_rate)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Activation":
                    line += 1
                    activation_name = f[line].replace('\n','')
                    line += 1
                    parameters = eval(f[line].replace('\n',''))
                    line += 1
                    layer = Activation(activation_name, parameters=parameters)
                    self.layers.append(layer)
                elif f[line].replace('\n','') == "Reshape":
                    line += 1
                    input_shape = eval(f[line].replace('\n',''))
                    line += 1
                    output_shape = eval(f[line].replace('\n',''))
                    line += 1
                    layer = Reshape(input_shape, output_shape)
                    self.layers.append(layer)
                elif f[line] == "END":
                    break
                else:
                    print(line)
                    line += 1
        w.close
        print(f"Model loaded from {filename}")