# this is to represent knowledge in a computer using or and and statements

class Symbol():
    symbols = []
    symNo = 0
    def __init__(self,name,*args):
        self.name = name
        if len(args) != 0:
            self.description = args[0]
        self.state = False
        Symbol.symbols.append(self)
        Symbol.symNo += 1

def Or(*args):
    value = False
    for arg in args:
        if isinstance(arg,Symbol):
            value = value or arg.state
        else:
            value = value or arg
    return value

def And(*args):
    value = True
    for arg in args:
        if isinstance(arg,Symbol):
            value = value and arg.state
        else:
            value = value and arg
    return value