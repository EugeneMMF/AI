from logic import *

mustard = Symbol("mustard")
plum = Symbol("plum")
scarlet = Symbol("scarlet")

ballroom = Symbol("ballroom")
kitchen = Symbol("kitchen")
library = Symbol("library")

knife = Symbol("knife")
revolver = Symbol("revolver")
wrench = Symbol("wrench")

KB = KnowledgeBase(
    Or(mustard, plum, scarlet),
    Or(ballroom, kitchen, library),
    Or(knife, revolver, wrench)
)

def check_knowledge(KB):
    for symbol in Symbol.allSymbols:
        if checkModel(KB,symbol):
            print(symbol.name,": YES")
        elif not checkModel(KB,Not(symbol)):
            print(symbol.name,": MAYBE")


KB.add(Not(mustard))
KB.add(Not(kitchen))
KB.add(Not(revolver))
KB.add(Or(Not(scarlet),Not(library),Not(wrench)))
KB.add(Not(plum))
KB.add(Not(ballroom))

# print(KB.formula())
check_knowledge(KB)