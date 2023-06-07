import sys
from time import time
from logic import *

characters = ["Gilderoy","Minerva","Pomona","Horace"]
houses = ["Gryffindor","Hufflepuff","Ravenclaw","Slytherin"]

symbols = []
for character in characters:
    for house in houses:
        symbols.append(Symbol(f"{character}{house}"))

knowledge = And()
for character in characters:
    orr = Or()
    for house in houses:
        orr.add(Symbol(f"{character}{house}"))
    knowledge.add(orr)

for character in characters:
    for h1 in houses:
        for h2 in houses:
            if h1 != h2:
                knowledge.add(
                    Implication(Symbol(f"{character}{h1}"),Not(Symbol(f"{character}{h2}")))
                )

for house in houses:
    for c1 in characters:
        for c2 in characters:
            if c1 != c2:
                knowledge.add(
                    Implication(Symbol(f"{c1}{house}"),Not(Symbol(f"{c2}{house}")))
                )

knowledge.add(
    Or(Symbol("GilderoyGryffindor"),Symbol("GilderoyHufflepuff"))
)

knowledge.add(
    Not(Symbol("PomonaSlytherin"))
)

knowledge.add(
    Symbol("MinervaGryffindor")
)

# print(knowledge.toString())

# print(id(symbols[0]),symbols[0].toString())
# print((symbols[0])==Symbol("GilderoyGryffindor"))
# print(id(Symbol("GilderoyGryffindor")),Symbol("GilderoyGryffindor").toString())

a1 = 0
a2 = 0

print("###################   By Enumeration   ############################")
start = time()
for symbol in symbols:
    if checkModelByEnumeration(knowledge, symbol):
        print(symbol.toString())
        pass
stop = time()
print("Enumeration runtime:",stop - start,"\bs")

print("###################   By Resolution    ############################")
start = time()
for symbol in symbols:
    if checkModelByResolution(knowledge, symbol):
        print(symbol.toString())
        pass
stop = time()
print("Resolution runtime:",stop - start,"\bs")