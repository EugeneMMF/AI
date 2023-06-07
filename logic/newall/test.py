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
        orr.add(Symbol(character + house))
    knowledge.add(orr)

for character in characters:
    for h1 in houses:
        for h2 in houses:
            if h1 != h2:
                knowledge.add(
                    Implication(Symbol(character + h1),Symbol(character + h2))
                )

for house in houses:
    for c1 in characters:
        for c2 in characters:
            if c1 != c2:
                knowledge.add(
                    Implication(Symbol(c1 + house),Symbol(c2 + house))
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

print(id(symbols[0]),symbols[0].toString())
print(id(Symbol("GilderoyGryffindor")),Symbol("GilderoyGryffindor").toString())


# for symbol in symbols:
#     if checkModelEnumeration(knowledge, symbol):
#         print(symbol.toString())
