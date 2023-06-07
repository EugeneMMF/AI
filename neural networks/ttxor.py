import sys
from neural_net import *

x_train = [
    matrix([0,0],2),
    matrix([0,1],2),
    matrix([1,0],2),
    matrix([1,1],2),
]

y_train = [
    matrix(0,2),
    matrix(1,2),
    matrix(1,2),
    matrix(0,2)
]

net = Network(
    DenseLayer(2,3),
    ActivationLayer(tanh,der_tanh),
    DenseLayer(3,1),
    ActivationLayer(tanh,der_tanh),
    loss = mse,
    loss_derivative = der_mse
)

net.fit(x_train,y_train,1000,0.1)

print(net.predict(x_train))