class ConstantSymbol():
    def __init__(self,name):
        self.name = name
        self.characteristic = ""

class PredicateSymbol():
    def __init__(self,name):
        self.name = name
    
    def change(self,symbol):
        symbol.characteristic = self.name

    __call__ = change

class ForAll():
    def __init__(self):
        