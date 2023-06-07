from neural_net import *
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def preprocess(x,y,limit):
    def to_3d(matrix):
        result = []
        result.append(matrix)
        return result
    zero_index = np.where(y == 1)[0][:limit]
    one_index = np.where(y == 0)[0][:limit]
    all_indices = np.hstack((zero_index,one_index))
    all_indices = np.random.permutation(all_indices)
    x,y = x[all_indices],y[all_indices]
    x_r = []
    for i in range(len(x)):
        x_r.append(to_3d(matrix_scalar_division(255,x[i])))
    result = []
    for i in range(len(y)):
        result.append(list(list(0 for j in range(2)) for k in range(1)))
        result[i][0][y[i]] = 1
    # print(x_r[0],y[0],result[0])
    return x_r,result

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,y_train = preprocess(x_train,y_train,50)
x_test,y_test = preprocess(x_test,y_test,5)

print("loaded")

net = Network(
    ConvolutionLayer((1,28,28),2,6),
    # ActivationLayer(relu,der_relu),
    ActivationLayer(tanh,der_tanh),
    PoolingLayer(3,'max'),
    # PoolingLayer(3,'max'),
    # ReshapeLayer((6,3,3),(1,54)),
    ReshapeLayer((6,9,9),(1,486)),
    DenseLayer(486,50),
    # ActivationLayer(relu,der_relu),
    ActivationLayer(tanh,der_tanh),
    DenseLayer(50,2),
    # ActivationLayer(relu,der_relu),
    # ActivationLayer(tanh,der_tanh),
    ActivationLayer(tanh,der_tanh),
    # SoftMaxLayer(),
    # ActivationLayer(sigmoid,der_sigmoid),
    # DenseLayer(10,2),
    # ReshapeLayer((1,28,28),(1,28*28)),
    # DenseLayer(784,100),
    # ActivationLayer(sigmoid,der_sigmoid),
    # DenseLayer(100,10),
    # ActivationLayer(sigmoid,der_sigmoid),
    loss = mse,
    loss_derivative = der_mse
)

net.fit(x_train,y_train,5,0.2)
result = net.predict(x_test)
print(result)
print(y_test)