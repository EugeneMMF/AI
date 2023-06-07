import copy

# class to represent symbols
class Symbol():
    allSymbols = []
    def __init__ (self,name):
        self.name = name
        Symbol.allSymbols.append(self.name)
    
    # convert the symbol to a string by return in the name of the symbol
    def toString(self):
        return self.name
    
    # create a hash for the symbol that is based on the name of the symbol
    # for equality to hold the hashes must be the same
    def __hash__(self):
        return hash(("symbol", self.name))
    
    # check for equality by comparing the names of the symbols and if both are symbols
    def __eq__(self, other):
        return isinstance(other,Symbol) and self.name == other.name
    
    # function to get the symbols in the symbol thus returns the name of the symbol
    def symbols(self):
        return {self.name}
    
    # function to evaluate the symbol against the model thus returns true or false based 
    # on the value assigned to the symbol in the model
    def evaluate(self, model):
        try:
            return bool(model[self.name])
        except:
            return Exception("not reached")

# class to represent "not" operation
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
    
    def evaluate(self, model):
        return bool(not self.operand.evaluate(model))
    
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
    
    def evaluate(self,model):
        return all(conjunct.evaluate(model) for conjunct in self.conjuncts)

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
    
    def evaluate(self, model):
        return any(disjunct.evaluate(model) for disjunct in self.disjuncts)

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

    def evaluate(self, model):
        return bool(not self.antecedent.evaluate(model) or self.consequent.evaluate(model))
    
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
    
    def evaluate(self, model):
        return bool(
            (not self.left.evaluate(model) or self.right.evaluate(model)) and
            (not self.right.evaluate(model) or self.left.evaluate(model))
        )

def checkModelByEnumeration(knowledge, query):
    def checkModel(knowledge, query, symbols, model):
        if not symbols:
            if knowledge.evaluate(model):
                return query.evaluate(model)
            else:
                return True
        else:
            remaining = symbols.copy()
            symbol = remaining.pop()
            modelT = model.copy()
            modelT[symbol] = True
            modelF = model.copy()
            modelF[symbol] = False
            return (checkModel(knowledge, query, remaining, modelT) and checkModel(knowledge, query, remaining, modelF))
    symbols = set.union(knowledge.symbols(),query.symbols())
    return checkModel(knowledge, query, symbols, {})

def checkModelByResolution(kb, qry):
    def removeImplications(clause):
        if isinstance(clause, And):
            i = 0
            while i < len(clause.conjuncts):
                clause.conjuncts[i] = removeImplications(clause.conjuncts[i])
                i += 1
            return clause
        elif isinstance(clause, Or):
            i = 0
            while i < len(clause.disjuncts):
                clause.disjuncts[i] = removeImplications(clause.disjuncts[i])
                i += 1
            return clause
        elif isinstance(clause, Symbol):
            return clause
        elif isinstance(clause, Not):
            clause.operand = removeImplications(clause.operand)
            return clause
        elif isinstance(clause, Implication):
            clause.antecedent = removeImplications(clause.antecedent)
            clause.consequent = removeImplications(clause.consequent)
            return Or(Not(clause.antecedent),clause.consequent)
        elif isinstance(clause, Biconditional):
            clause.left = removeImplications(clause.left)
            clause.right = removeImplications(clause.right)
            return And(
                Or(Not(clause.left), clause.right),
                Or(Not(clause.right), clause.left)
            )
    
    def arrangeNots(clause):
        if isinstance(clause, Symbol):
            return clause
        elif isinstance(clause, Not):
            if isinstance(clause.operand, Symbol):
                return clause
            elif isinstance(clause.operand, Not):
                return arrangeNots(clause.operand.operand)
            elif isinstance(clause.operand, And):
                i = 0
                newOr = Or()
                while i < len(clause.operand.conjuncts):
                    newOr.add(arrangeNots(Not(clause.operand.conjuncts[i])))
                    i += 1
                return newOr
            elif isinstance(clause.operand, Or):
                i = 0
                newAnd = And()
                while i < len(clause.operand.disjuncts):
                    newAnd.add(arrangeNots(Not(clause.operand.disjuncts[i])))
                    i += 1
                return newAnd
        elif isinstance(clause, And):
            i = 0
            while i < len(clause.conjuncts):
                clause.conjuncts[i] = arrangeNots(clause.conjuncts[i])
                i += 1
            return clause
        elif isinstance(clause, Or):
            i = 0
            while i < len(clause.disjuncts):
                clause.disjuncts[i] = arrangeNots(clause.disjuncts[i])
                i += 1
            return clause

    def fixArrangement(clause):
        if isinstance(clause, Symbol):
            return clause
        elif isinstance(clause, Not):
            return clause
        elif isinstance(clause, And):
            if len(clause.conjuncts) == 1:
                return clause.conjuncts[0]
            i = 0
            while i < len(clause.conjuncts):
                clause.conjuncts[i] = fixArrangement(clause.conjuncts[i])
                i += 1
            i = 0
            while i < len(clause.conjuncts):
                if isinstance(clause.conjuncts[i], And):
                    for conj in clause.conjuncts[i].conjuncts:
                        clause.add(conj)
                    clause.conjuncts.pop(i)
                    continue
                i += 1
            newAnd = And()
            for conj in clause.conjuncts:
                newAnd.add(conj)
            if len(newAnd.conjuncts) == 1:
                return newAnd.conjuncts[0]
            return newAnd
        elif isinstance(clause, Or):
            if len(clause.disjuncts) == 1:
                return clause.disjuncts[0]
            i = 0
            while i < len(clause.disjuncts):
                clause.disjuncts[i] = fixArrangement(clause.disjuncts[i])
                i += 1
            i = 0
            while i < len(clause.disjuncts):
                if isinstance(clause.disjuncts[i], Or):
                    for disj in clause.disjuncts[i].disjuncts:
                        clause.add(disj)
                    clause.disjuncts.pop(i)
                    continue
                i += 1
            newOr = Or()
            for disj in clause.disjuncts:
                newOr.add(disj)
            if len(newOr.disjuncts) == 1:
                return newOr.disjuncts[0]
            return newOr

    # puts clause into conjunctive normal form
    # clause must have already been removed implications and biconditionals
    # clause must have nots, if any, with operands that are symbols only
    def cnf(clause):
        # first fix the arrangement
        clause = fixArrangement(clause)
        # if clause is a symbol return it
        if isinstance(clause, Symbol):
            return clause
        # if clause is a not return it
        if isinstance(clause, Not):
            return clause
        # if clause is an and go through all its conjunctions 
        # converting them into conjunctive normal form and 
        # return the and object afterwards
        if isinstance(clause, And):
            i = 0
            while i < len(clause.conjuncts):
                clause.conjuncts[i] = cnf(clause.conjuncts[i])
                i += 1
            cl1 = copy.deepcopy(clause)
            clause = fixArrangement(clause)
            if cl1 == clause:
                return clause
            else:
                return cnf(clause)
        # Or's must be analysed
        if isinstance(clause, Or):
            i = 0
            cl = copy.deepcopy(clause)
            # go through all the disjunctions
            while i < len(clause.disjuncts):
                # if the disjunction is an And then we update it
                if isinstance(clause.disjuncts[i], And):
                    newAnd = And()
                    j = 0
                    # go through the disjunctions from scratch and pick a different disjunction 
                    # to use to resolve the current And disjunction
                    while j < len(clause.disjuncts):
                        if i != j:
                            # if the disjuntion we pick is a symbol or a not
                            # we add disjuntions of a pair of the symbol or not we have found and a conjunctive from the conjunction we had
                            # these are added to the new And statement
                            if isinstance(clause.disjuncts[j], Symbol) or isinstance(clause.disjuncts[j], Not):
                                for conj1 in clause.disjuncts[i].conjuncts:
                                    newAnd.add(Or(conj1,clause.disjuncts[j]))
                                if i < j:
                                    clause.disjuncts.pop(j)
                                    clause.disjuncts[i] = newAnd
                                    i = -1
                                    break
                                else:
                                    clause.disjuncts.pop(i)
                                    clause.disjuncts[j] = newAnd
                                    i = -1
                                    break
                            if isinstance(clause.disjuncts[j], And):
                                for conj1 in clause.disjuncts[i].conjuncts:
                                    for conj2 in clause.disjuncts[j].conjuncts:
                                        newAnd.add(Or(conj1,conj2))
                                if i < j:
                                    clause.disjuncts.pop(j)
                                    clause.disjuncts[i] = newAnd
                                    i = -1
                                    break
                                else:
                                    clause.disjuncts.pop(i)
                                    clause.disjuncts[j] = newAnd
                                    i = -1
                                    break
                        j += 1
                i += 1
            i = 0
            cl1 = copy.deepcopy(clause)
            clause = fixArrangement(clause)
            while cl1 != clause:
                cl1 = copy.deepcopy(clause)
                clause = fixArrangement(clause)
            if clause == cl:
                return clause
            return cnf(clause)

    def deduction(knowledge):
        explored = set()
        if isinstance(knowledge, And):
            i = 0
            while i < len(knowledge.conjuncts):
                if isinstance(knowledge.conjuncts[i], Symbol) or isinstance(knowledge.conjuncts[i], Not):
                    if not i in explored:
                        explored.add(i)
                        comparator = arrangeNots(Not(knowledge.conjuncts[i]))
                        j = 0
                        while j < len(knowledge.conjuncts):
                            if j != i:
                                if knowledge.conjuncts[j] == comparator:
                                    return True
                                elif isinstance(knowledge.conjuncts[j], Or):
                                    k = 0
                                    newOr = Or()
                                    while k < len(knowledge.conjuncts[j].disjuncts):
                                        if knowledge.conjuncts[j].disjuncts[k] == comparator:
                                            pass
                                        else:
                                            newOr.add(knowledge.conjuncts[j].disjuncts[k])
                                        k += 1
                                    if len(newOr.disjuncts) != len(knowledge.conjuncts[j].disjuncts):
                                        if len(newOr.disjuncts) == 1:
                                            knowledge.add(newOr.disjuncts[0])
                                        else:
                                            knowledge.add(newOr)
                            j += 1
                i += 1

    knowledge = copy.deepcopy(kb)
    query = copy.deepcopy(qry)
    if isinstance(knowledge, And):
        knowledge.add(Not(query))
    else:
        knowledge = And(knowledge,Not(query))
    knowledge = removeImplications(knowledge)
    knowledge = arrangeNots(knowledge)
    knowledge = fixArrangement(knowledge)
    knowledge = cnf(knowledge)
    return deduction(knowledge)