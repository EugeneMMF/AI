import logic
import copy

def alleq(value,iterable):
    for it in iterable:
        if it != value:
            return False
    return True

def anyeq(value,iterable):
    for it in iterable:
        if it == value:
            return True
    return False

class Symbol():
    allSymbols = []
    def __init__ (self,name):
        self.name = name
        Symbol.allSymbols.append(self.name)
    
    def toString(self):
        return self.name
    
    def __hash__(self):
        return hash(("symbol", self.name))
    
    def __eq__(self, other):
        return isinstance(other,Symbol) and self.name == other.name
    
    def symbols(self):
        return {self.name}
    
    def evaluate(self, args):
        return self
    
class Not():
    def __init__ (self,name):
        self.operand = name
    
    def toString(self):
        return f"¬({self.operand.toString()})"
    
    def __hash__(self):
        return hash(("not",hash(self.operand)))
    
    def __eq__(self, other):
        return isinstance(other,Not) and self.operand == other.operand
    
    def symbols(self):
        return self.operand.symbols()
    
    def evaluate(self, args):
        self.operand = self.operand.evaluate(args)
        return self
    
class And():
    def __init__(self, *conjuncts):
        self.conjuncts = list(conjuncts)
    
    def __eq__(self, other):
        if isinstance(other, And) and len(self.conjuncts) == len(other.conjuncts):
            for conj in self.conjuncts:
                if not conj in other.conjuncts:
                    return False
            return True
        else:
            return False
    
    def __hash__(self):
        return hash(
            ("and",tuple(hash(conjunct) for conjunct in self.conjuncts))
            )
    
    def toString(self):
        conjuncts = " ∧ ".join(conjunct.toString() for conjunct in self.conjuncts)
        return f"({conjuncts})"

    def add(self, conjunct):
        if isinstance(conjunct,list):
            for conj in conjunct:
                if not conj in self.conjuncts:
                    self.conjuncts.append(conj)
        else:
            if not conjunct in self.conjuncts:
                self.conjuncts.append(conjunct)
    
    def symbols(self):
        return set.union(*[conjunct.symbols() for conjunct in self.conjuncts])
    
    def evaluate(self, args):
        i = 0
        while i < len(self.conjuncts):
            self.conjuncts[i] = self.conjuncts[i].evaluate(args)
            i += 1
        return self

class Or():
    def __init__(self, *disjuncts):
        self.disjuncts = list(disjuncts)
    
    def __eq__(self, other):
        if isinstance(other, Or) and len(self.disjuncts) == len(other.disjuncts):
            for disj in self.disjuncts:
                if not disj in other.disjuncts:
                    return False
            return True
        else:
            return False
    
    def __hash__(self):
        return hash(
            ("or",tuple(hash(disjunct) for disjunct in self.disjuncts))
            )
    
    def toString(self):
        disjuncts = " ∨ ".join(disjunct.toString() for disjunct in self.disjuncts)
        return f"({disjuncts})"

    def add(self, disjunct):
        if isinstance(disjunct,list):
            for disj in disjunct:
                if not disj in self.disjuncts:
                    self.disjuncts.append(disj)
        else:
            if not disjunct in self.disjuncts:
                self.disjuncts.append(disjunct)
    
    def symbols(self):
        return set.union(*[disjunct.symbols() for disjunct in self.disjuncts])
    
    def evaluate(self, args):
        i = 0
        while i < len(self.disjuncts):
            self.disjuncts[i] = self.disjuncts[i].evaluate(args)
            i += 1
        return self

class Implication():
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def __eq__(self, other):
        return isinstance(other, Implication) and self.antecedent == other.antecedent and self.consequent == other.consequent
    
    def __hash__(self):
        return hash(
            ("implication",(hash(self.antecedent),hash(self.consequent)))
        )
    
    def toString(self):
        return f"({self.antecedent.toString()} => {self.consequent.toString()})"
    
    def symbols(self):
        return set.union(self.antecedent.symbols(),self.consequent.symbols())

    def evaluate(self, args):
        self.antecedent = self.antecedent.evaluate(args)
        self.consequent = self.consequent.evaluate(args)
        return self
    
class Biconditional():
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        return isinstance(other, Biconditional) and self.left == other.left and self.right == other.right
    
    def __hash__(self):
        return hash(
            ("biconditional",(hash(self.left),hash(self.right)))
        )
    
    def toString(self):
        return f"({self.left.toString()} <=> {self.right.toString()})"
    
    def symbols(self):
        return set.union(self.left.symbols(),self.right.symbols())
    
    def evaluate(self, args):
        self.left = self.left.evaluate(args)
        self.right = self.right.evaluate(args)
        return self

class ConstantSymbol():
    allSymbols = []
    def __init__(self, name):
        self.name = name
        ConstantSymbol.allSymbols.append(self)

    def evaluate(self, args):
        return self.name

class PredicateSymbol():
    def __init__(self, name):
        self.name = name
        self.values = []
    
    def called(self, *args):
        symb = PredicateSymbol(self.name)
        symb.values = tuple(args)
        return symb
    
    __call__ = called

    def evaluate(self, args):
        variables = copy.deepcopy(self.values)
        name = copy.deepcopy(self.name)
        for variable in variables:
            if isinstance(variable,ConstantSymbol):
                name += variable.evaluate(args)
            else:
                name += args[variable].evaluate(args)
        symb = Symbol(name)
        return symb

class ForAll():
    def __init__(self, variable, application):
        self.variable = variable
        self.application = application
    
    def evaluate(self, args):
        complete = And()
        constants = ConstantSymbol.allSymbols
        for constant in constants:
            temp = copy.deepcopy(self.application)
            newArg = {self.variable:constant}
            newArg.update(args)
            complete.add(temp.evaluate(newArg))
        return complete

class ThereExists():
    def __init__(self, variable, application):
        self.variable = variable
        self.application = application
    
    def evaluate(self, args):
        complete = Or()
        constants = ConstantSymbol.allSymbols
        for constant in constants:
            temp = copy.deepcopy(self.application)
            newArg = {self.variable:constant}
            newArg.update(args)
            complete.add(temp.evaluate(newArg))
        return complete

def checkFirstOrderLogicModel(kb, qry):
    def convertClass(sentence):
        if isinstance(sentence,And):
            newSentence = logic.And()
            i = 0
            while i < len(sentence.conjuncts):
                newSentence.add(convertClass(sentence.conjuncts[i]))
                i += 1
        elif isinstance(sentence,Or):
            newSentence = logic.Or()
            i = 0
            while i < len(sentence.disjuncts):
                newSentence.add(convertClass(sentence.disjuncts[i]))
                i += 1
        elif isinstance(sentence,Biconditional):
            newSentence = logic.Biconditional(convertClass(sentence.left),convertClass(sentence.right))
        elif isinstance(sentence,Implication):
            newSentence = logic.Implication(convertClass(sentence.antecedent),convertClass(sentence.consequent))
        elif isinstance(sentence,Not):
            newSentence = logic.Not(convertClass(sentence.operand))
        elif isinstance(sentence,Symbol):
            newSentence = logic.Symbol(sentence.name)
        return newSentence

    query = copy.deepcopy(qry)
    knowledge = copy.deepcopy(kb)
    query = query.evaluate({})
    knowledge = knowledge.evaluate({})
    knowledge = convertClass(knowledge)
    query = convertClass(query)
    return logic.checkModelByResolution(knowledge, query)