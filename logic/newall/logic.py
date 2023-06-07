###########################################################################################
####################################    LOGIC LIBRARY   ###################################
###########################################################################################

class Symbol():
    def __init__(self,name,*args):
        self.name = name
        self.description = ""
        self.state = False
        if len(args) != 0:
            self.description = args[0]

    def evaluate(self):
        return self.state
    
    def toString(self):
        return self.name
    
    def validate(self):
        return self

class Not():
    def __init__(self,symbol):
        self.symbol =  symbol

    def evaluate(self):
        return not self.symbol.evaluate()
    
    def toString(self):
        return f"Not({self.symbol.toString()})"
    
    def validate(self):
        self.symbol = self.symbol.validate()
        return self

class And():
    def __init__ (self, *args):
        self.symbols = []
        if len(args) != 0:
            for arg in args:
                if all(id(arg) != id(item) for item in self.symbols):
                    self.symbols.append(arg)
    
    def add(self,*args):
        if len(args) != 0:
            for arg in args:
                if isinstance(arg,list):
                    for item in arg:
                        if all(id(item) != id(it) for it in self.symbols):
                            self.symbols.append(item)
                else:
                    if all(id(arg) != id(it) for it in self.symbols):
                        self.symbols.append(arg)
    
    def evaluate(self):
        value = True
        for symbol in self.symbols:
            value = value and symbol.evaluate()
            if not value:
                return False
        return value
    
    def toString(self):
        myStr = "And("
        i = 0
        while i < len(self.symbols):
            myStr += self.symbols[i].toString()
            if i != len(self.symbols) - 1:
                myStr += " & "
            i += 1
        myStr += ")"
        return myStr
    
    def validate(self):
        if len(self.symbols) == 1:
            return self.symbols[0].validate()
        else:
            i = 0
            while i < len(self.symbols):
                self.symbols[i] = self.symbols[i].validate()
                i += 1
            return self
    
class Or():
    def __init__ (self, *args):
        self.symbols = []
        if len(args) != 0:
            for arg in args:
                if all(id(arg) != id(it) for it in self.symbols):
                    self.symbols.append(arg)
    
    def add(self,*args):
        if len(args) != 0:
            for arg in args:
                if isinstance(arg,list):
                    for item in arg:
                        if all(id(item) != id(it) for it in self.symbols):
                            self.symbols.append(item)
                else:
                    if all(id(arg) != id(it) for it in self.symbols):
                        self.symbols.append(arg)
    
    def evaluate(self):
        value = False
        for symbol in self.symbols:
            value = value or symbol.evaluate()
            if value:
                return True
        return value
    
    def toString(self):
        myStr = "Or("
        i = 0
        while i < len(self.symbols):
            myStr += self.symbols[i].toString()
            if i != len(self.symbols) - 1:
                myStr += " | "
            i += 1
        myStr += ")"
        return myStr
        
    def validate(self):
        if len(self.symbols) == 1:
            return self.symbols[0].validate()
        else:
            i = 0
            while i < len(self.symbols):
                self.symbols[i] = self.symbols[i].validate()
                i += 1
            return self

class Implication():
    def __init__(self,this,that):
        self.this = this
        self.that = that

    def evaluate(self):
        Or(Not(self.this),self.that).evaluate()
    
    def toString(self):
        return f"Implication({self.this.toString()} --> {self.that.toString()})"
    
    def validate(self):
        self.this = self.this.validate()
        self.that = self.that.validate()
        return self

class Biconditional():
    def __init__(self,this,that):
        self.this = this
        self.that = that

    def evaluate(self):
        And(Or(Not(self.this),self.that),Or(self.this,Not(self.that))).evaluate()
    
    def toString(self):
        return f"Biconditional({self.this.toString()} <--> {self.that.toString()})"
    
    def validate(self):
        self.this = self.this.validate()
        self.that = self.that.validate()
        return self

class Knowledge():
    def __init__(self,*args):
        self.symbols = []
        for arg in args:
            if all(id(arg) != id(item) for item in self.symbols):
                self.symbols.append(arg)
    
    def add(self,*args):
        if len(args) != 0:
            for arg in args:
                if isinstance(arg,list):
                    for item in arg:
                        if all(id(item) != id(it) for it in self.symbols):
                            self.symbols.append(item)
                else:
                    if all(id(arg) != id(it) for it in self.symbols):
                        self.symbols.append(arg)
    
    def evaluate(self):
        fullAnd = And()
        for item in self.symbols:
            fullAnd.add(item)
        return fullAnd.evaluate()

    def toString(self):
        myStr = "Knowledge("
        for item in self.symbols:
            myStr += item.toString()
            if id(item) != id(self.symbols[-1]):
                myStr += " , "
        myStr += ")"
        return myStr

    def validate(self):
        if len(self.symbols) == 1:
            return self.symbols[0].validate()
        else:
            i = 0
            while i < len(self.symbols):
                self.symbols[i] = self.symbols[i].validate()
                i += 1
            return self

def generateWorlds(start,end):
    possibilities = []
    if start < end:
        returned = generateWorlds(start + 1, end)
        for return1 in returned:
            possibilities.append("0" + return1)
            possibilities.append("1" + return1)
    else:
        return ["1","0"]
    return possibilities

def getSymbols(statement):
    allSymbols = []
    if isinstance(statement,Knowledge) or isinstance(statement,And) or isinstance(statement,Or):
        for symbol in statement.symbols:
            returned = getSymbols(symbol)
            for return1 in returned:
                if all(id(return1) != id(item) for item in allSymbols):
                    allSymbols.append(return1)
    elif isinstance(statement,Implication) or isinstance(statement,Biconditional):
        returned = getSymbols(statement.this)
        for return1 in returned:
            if all(id(return1) != id(item) for item in allSymbols):
                allSymbols.append(return1)
        returned = getSymbols(statement.that)
        for return1 in returned:
            if all(id(return1) != id(item) for item in allSymbols):
                allSymbols.append(return1)
    elif isinstance(statement,Not):
        returned = getSymbols(statement.symbol)
        for return1 in returned:
            if all(id(return1) != id(item) for item in allSymbols):
                allSymbols.append(return1)
    elif isinstance(statement,Symbol):
        return [statement]
    return allSymbols

def checkModelEnumeration(knowledge,query):
    knowledgeValues = []
    queryValues = []
    temporary = Knowledge(knowledge,query)
    allSymbols = getSymbols(temporary)
    allWorlds = generateWorlds(1,len(allSymbols))
    for world in allWorlds:
        i = 0
        while i < len(world):
            if world[i] == "1":
                allSymbols[i].state = True
            else:
                allSymbols[i].state = False
            i += 1
        queryValues.append(query.evaluate())
        knowledgeValues.append(knowledge.evaluate())
    i = 0 
    while i < len(queryValues):
        if knowledgeValues[i] and not queryValues[i]:
            return False
        i += 1
    return True

def removeImplicationsAndBiconditionals(clause):
    if isinstance(clause,Knowledge):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = removeImplicationsAndBiconditionals(clause.symbols[i])
            i += 1
    elif isinstance(clause,And):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = removeImplicationsAndBiconditionals(clause.symbols[i])
            i += 1
    elif isinstance(clause,Or):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = removeImplicationsAndBiconditionals(clause.symbols[i])
            i += 1
    elif isinstance(clause,Not):
        clause.symbol = removeImplicationsAndBiconditionals(clause.symbol)
    elif isinstance(clause,Implication):
        return Or(Not(removeImplicationsAndBiconditionals(clause.this)), removeImplicationsAndBiconditionals(clause.that))
    elif isinstance(clause,Biconditional):
        return And(
            Or(Not(removeImplicationsAndBiconditionals(clause.this)), removeImplicationsAndBiconditionals(clause.that)),
            Or(Not(removeImplicationsAndBiconditionals(clause.that)), removeImplicationsAndBiconditionals(clause.this))
        )
    elif isinstance(clause,Symbol):
        return clause
    return clause

def arrangeNots(clause):
    if isinstance(clause,Symbol):
        return clause
    elif isinstance(clause,Not):
        if isinstance(clause.symbol,Or):
            i = 0
            newAnd = And()
            while i < len(clause.symbol.symbols):
                newAnd.add(arrangeNots(Not(clause.symbol.symbols[i])))
                i += 1
            clause = newAnd
        elif isinstance(clause.symbol,And):
            i = 0
            newOr = Or()
            while i < len(clause.symbol.symbols):
                newOr.add(arrangeNots(Not(clause.symbol.symbols[i])))
                i += 1
            clause = newOr
        elif isinstance(clause.symbol,Not):
            clause = arrangeNots(clause.symbol.symbol)
        return clause
    elif isinstance(clause,And):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = arrangeNots(clause.symbols[i])
            i += 1
        return clause
    elif isinstance(clause,Or):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = arrangeNots(clause.symbols[i])
            i += 1
        return clause
    elif isinstance(clause,Knowledge):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = arrangeNots(clause.symbols[i])
            i += 1
        return clause
    elif isinstance(clause, Implication) or isinstance(clause, Biconditional):
        return arrangeNots(removeImplicationsAndBiconditionals(clause))

def rearrange(clause):
    if isinstance(clause, Or):
        i = 0
        while i < len(clause.symbols):
            if isinstance(clause.symbols[i], Or):
                for obj in clause.symbols[i].symbols:
                    clause.add(obj)
                clause.symbols.pop(i)
                continue
            i += 1
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = rearrange(clause.symbols[i])
            i += 1
    elif isinstance(clause, And):
        i = 0
        while i < len(clause.symbols):
            if isinstance(clause.symbols[i], And):
                for obj in clause.symbols[i].symbols:
                    clause.add(obj)
                clause.symbols.pop(i)
                continue
            i += 1
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = rearrange(clause.symbols[i])
            i += 1
    elif isinstance(clause, Knowledge):
        i = 0
        while i < len(clause.symbols):
            if isinstance(clause.symbols[i], And):
                for obj in clause.symbols[i].symbols:
                    clause.add(obj)
                clause.symbols.pop(i)
                continue
            i += 1
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = rearrange(clause.symbols[i])
            i += 1
    elif isinstance(clause, Not):
        clause.symbol = rearrange(clause.symbol)
    elif isinstance(clause, Implication) or isinstance(clause, Biconditional):
        clause = removeImplicationsAndBiconditionals(clause)
        clause = rearrange(clause)
    return clause

def orJoin(sen1, sen2):
    newAnd = And()
    if isinstance(sen1, And):
        i = 0
        while i < len(sen1.symbols):
            if isinstance(sen2, Symbol) or isinstance(sen2, Not):
                newAnd.add(Or(sen1.symbols[i],sen2))
            elif isinstance(sen2, And):
                j = 0
                while j < len(sen2.symbols):
                    newAnd.add(Or(sen1.symbols[i],sen2.symbols[j]))
                    j += 1
            i += 1
        return newAnd
            

def fixOrder(clause):
    if isinstance(clause, Or):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = fixOrder(clause.symbols[i])
            i += 1
        i = 0
        while i < len(clause.symbols):
            if isinstance(clause.symbols[i], And):
                j = 0
                while j < len(clause.symbols):
                    if i != j:
                       clause.symbols[i] = orJoin(clause.symbols[i],clause.symbols[j])
                       clause.symbols.pop(j)
                       i = -1
                       break
                    j += 1
            i += 1
    elif isinstance(clause,And) or isinstance(clause, Knowledge):
        i = 0
        while i < len(clause.symbols):
            clause.symbols[i] = fixOrder(clause.symbols[i])
            i += 1
    return clause.validate()

def getNotsSymbols(clause):
    symbols = []
    if isinstance(clause, Symbol) or isinstance(clause, Not):
        return [clause]
    else:
        for object in clause.symbols:
            returned = getSymbols(object)
            for ret in returned:
                symbols.append(ret)
        return symbols

def decision(clause):
    if isinstance(clause, Knowledge) or isinstance(clause, And):
        i = 0
        while i < len(clause.symbols):
            j = i + 1
            change = False
            while j < len(clause.symbols):
                print(clause.toString())
                symbi = getNotsSymbols(clause.symbols[i])
                for sy in symbi:
                    print("symbi = ",sy.toString())
                symbj = getNotsSymbols(clause.symbols[j])
                for sy in symbj:
                    print("symbj = ",sy.toString())
                k = 0
                while k < len(symbi):
                    l = 0
                    while l < len(symbj):
                        if isinstance(symbi[k],Not):
                            if id(arrangeNots(Not(symbi[k]))) == id(symbj[l]):
                                symbi.pop(k)
                                symbj.pop(l)
                                change = True
                                break
                        elif isinstance(symbj[l],Not):
                            if id(arrangeNots(Not(symbj[l]))) == id(symbi[k]):
                                symbi.pop(k)
                                symbj.pop(l)
                                change = True
                                break
                        l += 1
                    if change:
                        totalLength = len(symbi) + len(symbj)
                        if len(symbj) > 1:
                            clause.symbols.pop(j)
                        if len(symbi) > 1:
                            clause.symbols.pop(i)
                        if totalLength == 0:
                            return True
                        elif totalLength == 1:
                            if len(symbi) == 0:
                                clause.add(symbj[0])
                            else:
                                clause.add(symbi[0])
                        else:
                            newOr = Or()
                            newOr.add(symbi)
                            newOr.add(symbj)
                            clause.add(newOr)
                        break
                    k += 1
                # if change:
                #     j = 0
                #     i = 0
                #     break
                j += 1
            # if change:
            #     i = 0
            #     j = 0
            #     continue
            i += 1
        return False

def conjunctiveNormalForm(clause):
    clause = removeImplicationsAndBiconditionals(clause)
    clause = arrangeNots(clause)
    clause = rearrange(clause)
    clause = fixOrder(clause)
    clause = rearrange(clause)
    print(clause.toString())
    return decision(clause)

def checkModelResolution(knowledge,query):
    temporary = knowledge
    temporary.add(Not(query))
    return conjunctiveNormalForm(temporary)