from knowledge import *

jane = Symbol('jane',"Jane is alive")
eric = Symbol('eric')



for i in range(1):
    Symbol.symbols[i].state = True
    for j in range(Symbol.symNo):
        if i != j:
            Symbol.symbols[j].state = True
            print(And(jane,Or(jane,eric)))
            Symbol.symbols[j].state = False
            print(And(jane,Or(jane,eric)))
    Symbol.symbols[i].state = False
    for j in range(Symbol.symNo):
        if i != j:
            Symbol.symbols[j].state = True
            print(And(jane,Or(jane,eric)))
            Symbol.symbols[j].state = False
            print(And(jane,Or(jane,eric)))
print(jane.description)

