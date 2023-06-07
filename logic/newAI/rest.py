from fologic import *

characters = ["Gilderoy","Minerva","Pomona","Horace"]
houses = ["Gryffindor","Hufflepuff","Ravenclaw","Slytherin"]

person = PredicateSymbol("Person")
house = PredicateSymbol("House")
minerva = ConstantSymbol("Minerva")
horace = ConstantSymbol("Horace")
pomona = ConstantSymbol("Pomona")
gilderoy = ConstantSymbol("Gilderoy")
gryffindor = ConstantSymbol("Gryffindor")
slytherin = ConstantSymbol("Slytherin")
hufflepuff = ConstantSymbol("Hufflepuff")
ravenclaw = ConstantSymbol("Ravenclaw")
belongsto = PredicateSymbol("BelongsTo")

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

print(knowledge.evaluate({}).toString())