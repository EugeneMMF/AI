from logic import *

colors = ["red","blue","green","yellow"]
positions = ['1','2','3','4']

symbols = []

# create the symbols
for position in positions:
    for color in colors:
        symbols.append(Symbol(f"{color}{position}"))

# create the knowledge
knowledge = And()

# start adding knowledge that a color in one position means it cannot be in another
for color in colors:
    for i in positions:
        for j in positions:
            if i != j:
                knowledge.add(Implication(Symbol(f"{color}{i}"),Not(Symbol(f"{color}{j}"))))

# add knowledge that a position having a color cannot have another color
for pos in positions:
    for i in colors:
        for j in colors:
            if i != j:
                knowledge.add(Implication(Symbol(f"{i}{pos}"),Not(Symbol(f"{j}{pos}"))))

# add knowledge that each position has a color
for pos in positions:
    statement = Or()
    for color in colors:
        statement.add(Symbol(f"{color}{pos}"))
    knowledge.add(statement)

# add knowledge that each color has a position
for color in colors:
    statement = Or()
    for pos in positions:
        statement.add(Symbol(f"{color}{pos}"))
    knowledge.add(statement)

# add knowledge from the first piece of evidence
evidence = ["red1","blue2","green3","yellow4"]
i = 0
statement = Or()
while i < len(evidence)-1:
    j = i + 1
    while j < len(evidence):
        statement.add(And(Symbol(evidence[i]),Symbol(evidence[j])))
        j += 1
    i += 1
knowledge.add(statement)

# add knowledge from the second piece of evidence
evidence = ["blue1","red2","green3","yellow4"]
for e in evidence:
    knowledge.add(Not(Symbol(e)))

# check for entailment of the symbols
for symbol in symbols:
    if checkModelByResolution(knowledge,symbol):
        print(symbol.toString(),"\b: YES")
    elif not checkModelByResolution(knowledge,Not(symbol)):
        print(symbol.toString(),"\b: MAYBE")