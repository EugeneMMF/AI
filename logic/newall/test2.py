from logic2 import *

person = PredicateSymbol("person")
house = PredicateSymbol("house")

jane = ConstantSymbol("jane")

person(jane)
house(jane)

print(jane.characteristic)