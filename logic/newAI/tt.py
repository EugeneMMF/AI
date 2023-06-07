import sys
from logic import *

characters = ["Gi","Mi","Po","Ho"]
houses = ["Gr","Hu","Ra","Sl"]

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
    Or(Symbol("GiGr"),Symbol("GiHu"))
)

knowledge.add(
    Not(Symbol("PoSl"))
)

knowledge.add(
    Symbol("MiGr")
)

# print(knowledge.toString())

# print(id(symbols[0]),symbols[0].toString())
# print((symbols[0])==Symbol("GilderoyGryffindor"))
# print(id(Symbol("GilderoyGryffindor")),Symbol("GilderoyGryffindor").toString())


for symbol in symbols:
    if checkModelByResolution(knowledge, symbol):
        print(symbol.toString())