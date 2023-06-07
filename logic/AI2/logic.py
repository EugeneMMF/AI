# This file allows you to solve propositional logic problems using both resolution and testing of all possible worlds
# checkModel(KB,query) uses model checking
# check_model(KB,query) uses resolution


# Symbol class
class Symbol():
    def __init__(self,name,*args):
        self.name = name
        self.state = False
        self.description = ""
        if len(args) != 0:
            self.description = args[0]
    
    def evaluate(self):
        return self.state

    def validate(self):
        return self

    def formula(self):
        return self.name

# Not class
class Not():
    def __init__(self,object):
        self.object = object
    
    def evaluate(self):
        return not self.object.evaluate()

    def validate(self):
        return self

    def formula(self):
        return f"Not({self.object.formula()})"

# Or class
class Or():
    def __init__(self,*args):
        self.objects = []
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,Or):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,Or):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)
    
    def add(self,*args):
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,Or):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,Or):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)

    def evaluate(self):
        value = False
        for object in self.objects:
            value = value or object.evaluate()
        return value
    
    def validate(self):
        if len(self.objects) == 1:
            self = self.objects[0].validate()
            return self
        else:
            i = 0
            newOr = []
            while i < len(self.objects):
                if isinstance(self.objects[i],Or):
                    for object in self.objects[i]:
                        newOr.append(object.validate())
                else:
                    if not any(id(self.objects[i].validate()) == id(obj) for obj in newOr):
                        newOr.append(self.objects[i].validate())
                i += 1
            if len(newOr) == 1:
                self = newOr[0]
                return self
            self = Or(newOr)
            return self

    def formula(self):
        myStr = "Or("
        i = 0
        while i < len(self.objects):
            myStr += self.objects[i].formula()
            if i != (len(self.objects) - 1):
                myStr += " | "
            i += 1
        myStr += ")"
        return myStr

# And class
class And():
    def __init__(self,*args):
        self.objects = []
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,And):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,And):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)
    
    def add(self,*args):
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,And):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,And):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)

    def evaluate(self):
        value = True
        for object in self.objects:
            value = value and object.evaluate()
        return value

    def validate(self):
        if len(self.objects) == 1:
            self = self.objects[0].validate()
            return self
        else:
            i = 0
            newAnd = []
            while i < len(self.objects):
                if isinstance(self.objects[i],And):
                    for object in self.objects[i]:
                        newAnd.append(object.validate())
                else:
                    if not any(id(self.objects[i].validate()) == id(obj) for obj in newAnd):
                        newAnd.append(self.objects[i].validate())
                i += 1
            if len(newAnd) == 1:
                self = newAnd[0]
                return self
            self = And(newAnd)
            return self

    def formula(self):
        myStr = "And("
        i = 0
        while i < len(self.objects):
            myStr += self.objects[i].formula()
            if i != (len(self.objects) - 1):
                myStr += " & "
            i += 1
        myStr += ")"
        return myStr

# Knowledge base class
class Knowledge():
    def __init__(self,*args):
        self.objects = []
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,And):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,And):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)
    
    def add(self,*args):
        for arg in args:
            if isinstance(arg,list):
                for a in arg:
                    if isinstance(a,And):
                        for o in a.objects:
                            if not any(id(o) == id(ch) for ch in self.objects):
                                self.objects.append(o)
                    else:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
            else:
                if isinstance(arg,And):
                    for a in arg.objects:
                        if not any(id(a) == id(ch) for ch in self.objects):
                            self.objects.append(a)
                else:
                    if not any(id(arg) == id(ch) for ch in self.objects):
                        self.objects.append(arg)

    def evaluate(self):
        value = True
        for object in self.objects:
            value = value and object.evaluate()
        return value

    def validate(self):
        i = 0
        while i < len(self.objects):
            self.objects[i] = self.objects[i].validate()
            i += 1
        return self

    def formula(self):
        myStr = "{ "
        i = 0
        while i < len(self.objects):
            myStr += self.objects[i].formula()
            if i != (len(self.objects) - 1):
                myStr += " & "
            i += 1
        myStr += " }"
        return myStr

# Implication class
class Implication():
    def __init__(self,this,implies):
        self.this = this
        self.implies = implies
    
    def evaluate(self):
        newObject = Or(Not(self.this),self.implies)
        return newObject.evaluate()

    def validate(self):
        self.this = self.this.validate()
        self.implies = self.implies.validate()
        return self

    def formula(self):
        return "(" + self.this.formula() + " --> " + self.implies.formula() + ")"

# Bicoditional class
class Biconditional():
    def __init__(self,this,that):
        self.this = this
        self.that = that
    
    def evaluate(self):
        newObject = And(Or(Not(self.this),self.that),Or(Not(self.that),self.this))
        return newObject.evaluate()

    def validate(self):
        self.this = self.this.validate()
        self.that = self.that.validate()
        return self

    def formula(self):
        return "(" + self.this.formula() + " <--> " + self.that.formula() + ")"

# gets all Symbol objects in the object and returns them as a list
def getSymbols(object):
    symbols = []
    if isinstance(object,Symbol):
        symbols.append(object)
    elif isinstance(object,Not):
        symbols.append(object.object)
    elif isinstance(object,And):
        for subObject in object.objects:
            additions = getSymbols(subObject)
            for addition in additions:
                symbols.append(addition)
    elif isinstance(object,Or):
        for subObject in object.objects:
            additions = getSymbols(subObject)
            for addition in additions:
                symbols.append(addition)
    elif isinstance(object,Knowledge):
        for subObject in object.objects:
            additions = getSymbols(subObject)
            for addition in additions:
                symbols.append(addition)
    elif isinstance(object,Implication):
        additions = getSymbols(object.this)
        for addition in additions:
            symbols.append(addition)
        additions = getSymbols(object.implies)
        for addition in additions:
            symbols.append(addition)
    elif isinstance(object,Biconditional):
        additions = getSymbols(object.this)
        for addition in additions:
            symbols.append(addition)
        additions = getSymbols(object.that)
        for addition in additions:
            symbols.append(addition)
    newSymbols = []
    for symbol in symbols:
        if len(newSymbols) == 0:
            newSymbols.append(symbol)
        else:
            if any(id(newSymbol) == id(symbol) for newSymbol in newSymbols):
                pass
            else:
                newSymbols.append(symbol)
    return newSymbols

# removes all biconditionals in the sentence replacing them with implications
def removeBiconditional(sentence):
    i = 0
    if isinstance(sentence,Symbol):
        return sentence
    elif isinstance(sentence,Not):
        return sentence
    elif isinstance(sentence,Biconditional):
        return And(Implication(sentence.this,sentence.that),Implication(sentence.that,sentence.this))
    elif isinstance(sentence,Implication):
        return sentence
    elif isinstance(sentence,Or):
        i = 0
        while i < len(sentence.objects):
            sentence.objects[i] = removeBiconditional(sentence.objects[i])
            i += 1
        return sentence
    elif isinstance(sentence,And):
        i = 0
        while i < len(sentence.objects):
            sentence.objects[i] = removeBiconditional(sentence.objects[i])
            i += 1
        return sentence

# removes all biconditionals and implications in the sentence
def removeImplication(sentence):
    i = 0
    if isinstance(sentence,Symbol):
        return sentence
    elif isinstance(sentence,Not):
        return sentence
    elif isinstance(sentence,Biconditional):
        return And(Or(Not(sentence.this),sentence.that),Or(Not(sentence.that),sentence.this))
    elif isinstance(sentence,Implication):
        return Or(Not(sentence.this),sentence.implies)
    elif isinstance(sentence,Or):
        i = 0
        while i < len(sentence.objects):
            sentence.objects[i] = removeImplication(sentence.objects[i])
            i += 1
        return sentence
    elif isinstance(sentence,And):
        i = 0
        while i < len(sentence.objects):
            sentence.objects[i] = removeImplication(sentence.objects[i])
            i += 1
        return sentence
    elif isinstance(sentence,Knowledge):
        i = 0
        while i < len(sentence.objects):
            sentence.objects[i] = removeImplication(sentence.objects[i])
            i += 1
        return sentence

# generates an array of strings of all combinations of  0s and 1s of length end - start 
def tempValues(start,end):
    i = start
    values = []
    if i == end:
        return ["0","1"]
    elif i < end:
        additions = tempValues(i + 1,end)
        for addition in additions:
            values.append("0"+addition)
        for addition in additions:
            values.append("1"+addition)
        i += 1
        return values

# checks whether knowledge entails the query by building all possible models and checking them
def checkModel(knowledge,query):
    temp = And(knowledge,query)
    knowledge = removeImplication(knowledge)
    symbols = getSymbols(temp)
    values = tempValues(1,len(symbols))
    knowledgeValues = []
    queryValues = []
    for value in values:
        i = 0
        while i < len(value):
            if value[i] == "1":
                symbols[i].state = True
            else:
                symbols[i].state = False
            i += 1
        knowledgeValues.append(knowledge.evaluate())
        queryValues.append(query.evaluate())
    i = 0
    while i < len(knowledgeValues):
        if knowledgeValues[i] and not queryValues[i]:
            return False
        i += 1
    return True

# applies De Morgan's laws to move nots inside until you just have Not(Symbol) and takes neither biconditionals nor implications
def nots(sentence):
    if isinstance(sentence,Symbol):
        return sentence
    elif isinstance(sentence,Not):
        if isinstance(sentence.object,Symbol):
            return sentence
        elif isinstance(sentence.object,Not):
            sentence = sentence.object.object
            sentence = nots(sentence)
            return sentence
        elif isinstance(sentence.object,And):
            i = 0
            newOr = []
            while i < len(sentence.object.objects):
                newOr.append(nots(Not(sentence.object.objects[i])))
                i += 1
            sentence = Or(newOr)
            return sentence
        elif isinstance(sentence.object,Or):
            i = 0
            newAnd = []
            while i < len(sentence.object.objects):
                newAnd.append(nots(Not(sentence.object.objects[i])))
                i += 1
            sentence = And(newAnd)
            return sentence
    elif isinstance(sentence,Knowledge):
        i = 0
        newKnowledge = []
        while i < len(sentence.objects):
            newKnowledge.append(nots(sentence.objects[i]))
            i += 1
        return Knowledge(newKnowledge)
    elif isinstance(sentence,And):
        i = 0
        newAnd = []
        while i < len(sentence.objects):
            newAnd.append(nots(sentence.objects[i]))
            i += 1
        return And(newAnd)
    elif isinstance(sentence,Or):
        i = 0
        newOr = []
        while i < len(sentence.objects):
            newOr.append(nots(sentence.objects[i]))
            i += 1
        return Or(newOr)

# applies the distributive law for Or(And()) to And(Or()) and takes neither implications nor biconditionals
def distribute(sentence):
    if isinstance(sentence,Symbol):
        return sentence
    elif isinstance(sentence,Not):
        return sentence
    elif isinstance(sentence,And):
        i = 0
        newAnd = []
        while i < len(sentence.objects):
            newAnd.append(distribute(sentence.objects[i]))
            i += 1
        return And(newAnd).validate()
    elif isinstance(sentence,Knowledge):
        i = 0
        newAnd = []
        while i < len(sentence.objects):
            newAnd.append(distribute(sentence.objects[i]))
            i += 1
        return Knowledge(newAnd).validate()
    elif isinstance(sentence,Or):
        objs = []
        for obj in sentence.objects:
            objs.append(getSymbols(obj))
        res = subPutObjects(objs)
        if isinstance(res[0],list):
            toret = And()
            for re in res:
                tor = Or()
                for r in re:
                    tor.add(r)
                toret.add(tor)
        else:
            toret = Or(res)
        return toret

# generates an array of all possible combinations of the objects of the objects of obj picking one from each object of ob at a time
def subPutObjects(objs):
    selector = 1
    temp = objs[0]
    newTemp = []
    while selector < len(objs):
        newTemp = []
        j = 0
        while j < len(temp):
            for obj in objs[selector]:
                newTemp.append([temp[j],obj])
            j += 1
        proved = False
        j = 0
        while j < len(newTemp):
            for ob in newTemp[j]:
                if isinstance(ob,list):
                    for o in ob:
                        newTemp[j].append(o)
                    newTemp[j].pop(0)
            j += 1
        temp = newTemp
        selector += 1
    return temp

# checks if KB entails query using contradiction method
def check_model(KB,query):
    if isinstance(KB,Knowledge):
        testKB = Knowledge()
        for obj in KB.objects:
            testKB.add(obj)
        testKB.add(Not(query))
        testKB = removeImplication(testKB)
        testKB = nots(testKB)
        testKB = distribute(testKB)
        i = 0
        while i < len(testKB.objects):
            j = 0
            while j < len(testKB.objects):
                ret = check_model(testKB.objects[j],testKB.objects[i])
                if isinstance(ret,dict):
                    return True
                elif isinstance(ret,bool):
                    pass
                elif id(ret) != id(testKB.objects[j]):
                    testKB.objects.append(ret)
                    testKB.objects.pop(j)
                    i = -1
                    break
                j += 1
            i += 1
        return False
    elif isinstance(KB,Or):
        newOr = []
        for obj in KB.objects:
            ret = check_model(obj,query)
            if isinstance(ret,bool):
                newOr.append(obj)
        if len(newOr) == 0:
            return {}
        elif len(newOr) == len(KB.objects):
            return KB
        else:
            return Or(newOr)
    elif isinstance(KB,Symbol):
        if id(KB) == id(nots(Not(query))):
            return {}
        else:
            return False
    elif isinstance(KB,Not):
        if id(query) == id(nots(Not(KB))):
            return {}
        else:
            return False