from logic import *

A = Symbol("A")
B = Symbol("B")
C = Symbol("C")

arr = [A,C,Or(A,B)]

D = Knowledge(arr)
print(check_model(D,C))
print(checkModel(D,C))
D = Knowledge(A,B,Or(A,C,D))
print(distribute(D).formula())