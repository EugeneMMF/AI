from random import random
from time import time


def sampling(value,distribution):
    i = 0
    keys = list(distribution.keys())
    a = distribution[keys[0]]
    while a < value:
        i += 1
        a += distribution[keys[i]]
    return keys[i]

distribution = {"none":0.7,"light":0.2,"heavy":0.1}
distribution2 = {"none": 0, "light": 0, "heavy": 0}

expectedTime = 0.001
learningRate = 0.05
cycles = 200000
number = 7623
for j in range(cycles):
    start = time()
    a = 0
    for i in range(int(number)):
        sample = random()
        distribution2[sampling(sample,distribution)] += 1
        a+=1
    for val in distribution2.keys():
        distribution2[val]/=number
    print(distribution2)
    stop = time()
    currentTime = stop - start
    print(number,currentTime)
    number = number + learningRate * ((number / currentTime) * (expectedTime - currentTime))