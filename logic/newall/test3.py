from logic import *

A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
D = Symbol("D")

KB = Knowledge(
    Not(And(A,B)),
    A,
    Not(B),
    C
)

print(checkModelEnumeration(KB,A))
print(checkModelResolution(KB,A))

class O():
    symbols = []
    def add(self,name):
        O.symbols.append(name)
        return O.symbols[-1]
    __call__ = add

