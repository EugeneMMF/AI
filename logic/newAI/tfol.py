import sys
from time import time
from fologic import *

characters = ["Gilderoy","Minerva","Pomona","Horace"]
houses = ["Gryffindor","Hufflepuff","Ravenclaw","Slytherin"]


person = PredicateSymbol("Person")
house = PredicateSymbol("House")
minerva = ConstantSymbol("Minerva")
horace = ConstantSymbol("Horace")
pomona = ConstantSymbol("Pomona")
gilderoy = ConstantSymbol("Gilderoy")
characters = [minerva,horace,pomona,gilderoy]
gryffindor = ConstantSymbol("Gryffindor")
slytherin = ConstantSymbol("Slytherin")
hufflepuff = ConstantSymbol("Hufflepuff")
ravenclaw = ConstantSymbol("Ravenclaw")
houses = [gryffindor,slytherin,hufflepuff,ravenclaw]
belongsto = PredicateSymbol("BelongsTo")

all = []
for char in characters:
    all.append(char)
for ho in houses:
    all.append(ho)

x = "x"
y = "y"

knowledge = And(
    person(gilderoy),
    person(pomona),
    person(minerva),
    person(horace),
    house(gryffindor),
    house(hufflepuff),
    house(slytherin),
    house(ravenclaw),
    ForAll(x,Implication(house(x),Not(person(x)))),
    ForAll(y,Implication(house(y),ThereExists(x,And(person(x),belongsto(x,y))))),
    ForAll(x,Implication(person(x),ThereExists(y,And(house(y),belongsto(x,y))))),
    ForAll(y,Implication(house(y),Not(ThereExists(x,And(house(x),belongsto(x,y)))))),
    ForAll(x,Implication(person(x),Not(ThereExists(y,And(person(y),belongsto(x,y)))))),
    Or(belongsto(gilderoy,gryffindor),belongsto(gilderoy,hufflepuff)),
    Not(belongsto(pomona,slytherin)),
    belongsto(minerva,gryffindor)
)

for char in all:
    for other in all:
        if other.name != char.name:
            knowledge.add(
                ForAll(x,Implication(belongsto(x,char),Not(belongsto(x,other))))
            )



# print(knowledge.evaluate({}).toString())

symbols = []

for character in characters:
    for house in houses:
        symbols.append(belongsto(character,house))

a = 0
for symbol in symbols:
    start = time()
    print("checking",a, symbol.evaluate({}).toString())
    a+=1
    if checkFirstOrderLogicModel(knowledge,symbol):
        print((symbol).evaluate({}).toString())
    print(time()-start)
    # sys.exit()