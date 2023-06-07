class Symbol():
    allSymbols = []
    symbolNumber = 0
    def __init__(self,name,*args):
        self.name = name
        self.state = False
        if len(args) != 0:
            self.desciption = args[0]
        else:
            self.desciption = None
        Symbol.allSymbols.append(self)
        Symbol.symbolNumber += 1
    
    def evaluate(self):
        return self.state

    def formula(self):
        return self.name

class And():
    def __init__(self,*args):
        self.allObjects = []
        self.objectNumber = 0
        for arg in args:
            if isinstance(arg,list):
                # print("is list")
                for arr in arg:
                    self.allObjects.append(arr)
                    self.objectNumber += 1
            else:
                self.allObjects.append(arg)
                self.objectNumber += 1
    
    def add(self,*args):
        for arg in args:
            self.allObjects.append(arg)
            self.objectNumber += 1
    
    def evaluate(self):
        value = True
        for object in self.allObjects:
            if isinstance(object,Symbol):
                value = value and object.state
            else:
                value = value and object.evaluate()
        return value

    def formula(self):
        myStr = "And("
        for object in self.allObjects:
            myStr += object.formula()
            if object != self.allObjects[len(self.allObjects)-1]:
                myStr += ' & '
            else:
                myStr += ')'
        return myStr



class Or():
    def __init__(self,*args):
        self.allObjects = []
        self.objectNumber = 0
        for arg in args:
            if isinstance(arg,list):
                # print("is list")
                for arr in arg:
                    self.allObjects.append(arr)
                    self.objectNumber += 1
            else:
                self.allObjects.append(arg)
                self.objectNumber += 1
    
    def add(self,*args):
        for arg in args:
            self.allObjects.append(arg)
            self.objectNumber += 1

    def evaluate(self):
        value = False
        for object in self.allObjects:
            if isinstance(object,Symbol):
                value = value or object.state
            else:
                value = value or object.evaluate()
        return value
    
    def formula(self):
        myStr = "Or("
        for object in self.allObjects:
            myStr += object.formula()
            if object != self.allObjects[len(self.allObjects)-1]:
                myStr += ' v '
            else:
                myStr += ')'
        return myStr

class Not():
    def __init__(self,object):
        self.object = object
    
    def evaluate(self):
        value = True
        if isinstance(self.object,Symbol):
            value = not self.object.state
        else:
            value = not self.object.evaluate()
        return value
    
    def formula(self):
        myStr = "!("
        myStr += self.object.formula()
        myStr += ')'
        return myStr

class Implication():
    def __init__(self,this,impliesThis):
        self.this = this
        self.impliesThis = impliesThis

    def evaluate(self):
        obj = Or(Not(self.this),self.impliesThis)
        return obj.evaluate()
    
    def formula(self):
        myStr = "("
        myStr += self.this.formula()
        myStr += " -> "
        myStr += self.impliesThis.formula()
        myStr += ")"
        return myStr

class Biconditional():
    def __init__(self,this,impliesThis):
        self.this = this
        self.impliesThis = impliesThis

    def evaluate(self):
        obj = And(Implication(self.this,self.impliesThis),Implication(self.impliesThis,self.this))
        return obj.evaluate()
    
    def formula(self):
        myStr = "("
        myStr += self.this.formula()
        myStr += " <--> "
        myStr += self.impliesThis.formula()
        myStr += ")"
        return myStr

class KnowledgeBase():
    def __init__(self,*args):
        self.allObjects = []
        self.objectNumber = 0
        for arg in args:
            self.allObjects.append(arg)
            self.objectNumber += 1
    
    def add(self,*args):
        for arg in args:
            self.allObjects.append(arg)
            self.objectNumber += 1
    
    def evaluate(self):
        if len(self.allObjects) == 1:
            obj = self.allObjects[0]
            if isinstance(object,Symbol):
                value = obj.state
            elif isinstance(obj,Or):
                value = obj.evaluate()
            elif isinstance(obj,Not):
                value = obj.evaluate()
            elif isinstance(obj,Implication):
                value = obj.evaluate()
            return value
        else:
            obj = And(self.allObjects)
            # print(len(self.allObjects))
            value = obj.evaluate()
            return value
    
    def formula(self):
        myStr = "("
        for object in self.allObjects:
            myStr += object.formula()
            if object != self.allObjects[len(self.allObjects)-1]:
                myStr += ' & '
            else:
                myStr += ')'
        return myStr

    def formulaObjects(self):
        for object in self.allObjects:
            print(object.formula())

def generate(i,number):
    old = ["0","1"]
    new = []
    if i < number:
        i+=1
        result = generate(i,number)
        # print(result)
        for r in result:
            new.append(r+"0")
            new.append(r+"1")
    if len(new) == 0:
        return old
    else:
        return new

def checkModel (kb,query):
    results = []
    queryVal = []
    pval = []
    qval = []
    rval = []
    predictions = generate(1,Symbol.symbolNumber)
    for prediction in predictions:
        for i in range(len(prediction)):
            if prediction[i] == "1":
                Symbol.allSymbols[i].state = True
            else:
                Symbol.allSymbols[i].state = False
        queryVal.append(query.evaluate())
        results.append(kb.evaluate())
    # print("The kb gives:",results)
    # print("P value:",pval)
    # print("Q value:",qval)
    # print("R value:",rval)
    # print("Query value:",queryVal)
    for q in range(len(queryVal)):
        if results[q] and not queryVal[q]:
            return False
    return True