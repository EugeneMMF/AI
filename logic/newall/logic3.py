class Symbol():
    allSymbols = set()
    def __init__ (self,name):
        self.name = name
        Symbol.allSymbols.add(self)
    
class And():
    def __init__(self,*args):
        setattr(self.__class__,self._is_instance,False)
        if getattr(self.__class__,self._is_instance,False):
            raise NotImplemented
        self._is_instance = True