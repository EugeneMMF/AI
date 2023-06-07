from copy import deepcopy
from dis import dis
from operator import ne
from typing import Annotated
from logic import *

KB = KnowledgeBase()

A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
D = Symbol("D")

KB.add(Biconditional(Or(A,B),C))
KB = KnowledgeBase(Not(And(A,Or(B,Not(C),A))))
KB.add(And(A,B))

print(KB.formula())


# this part removes biconditionals in the knowledge base KB

def removeBiconditional(KB):
    while True:
        poped = True
        for i in range(len(KB.allObjects)):
            object = KB.allObjects[i]
            if isinstance(object,Biconditional):
                KB.add(Implication(object.this,object.impliesThis))
                KB.add(Implication(object.impliesThis,object.this))
                KB.allObjects.pop(i)
                KB.objectNumber = len(KB.allObjects)
                poped = False
        if poped:
            break
    return KB

KB = removeBiconditional(KB)
print(KB.formula())


# this part removes implications in the knowledge base KB

def removeImplications(KB):
    while True:
        poped = True
        for i in range(len(KB.allObjects)):
            object = KB.allObjects[i]
            if isinstance(object,Implication):
                KB.add(Or(Not(object.this), object.impliesThis))
                KB.allObjects.pop(i)
                KB.objectNumber = len(KB.allObjects)
                poped = False
        if poped:
            break
    return KB

KB = removeImplications(KB)
print(KB.formula())


# this part puts all Not operators  in KB on the inside up to a literal


def correct(object,*args):
    # print(object.formula(),args)
    if isinstance(object,Not):
        if isinstance(object.object,Symbol):
            if len(args) == 0:
                return correct(object.object,1)
            else:
                return correct(object.object)
        elif isinstance(object.object,Not):
            return object.object.object
        elif isinstance(object.object,And):
            return correct(object.object,1)
        elif isinstance(object.object,Or):
            return correct(object.object,1)
    elif isinstance(object,And):
        if len(args) == 0:
            newAnd = []
            for obj in object.allObjects:
                newAnd.append(correct(obj))
            return And(newAnd)
        else:
            newOr = []
            for obj in object.allObjects:
                newOr.append(correct((obj),1))
            return Or(newOr)
    elif isinstance(object,Or):
        if len(args) == 0:
            newOr = []
            for obj in object.allObjects:
                ot = correct(obj)
                newOr.append(ot)
            return Or(newOr)
        else:
            newAnd = []
            for obj in object.allObjects:
                newAnd.append(correct((obj),1))
            return And(newAnd)
    else:
        if len(args) == 0:
            return object
        else:
            return Not(object)

def notsInside(KB):
    newKB = ""
    while True:
        for i in range(len(KB.allObjects)):
            object = KB.allObjects[i]
            object = correct(object)
            KB.allObjects[i] = object
        # print(KB.formula())
        myStr = KB.formula()
        if newKB == myStr:
            break
        else:
            newKB = myStr
    return KB

KB = notsInside(KB)
print(KB.formula())
print(len(KB.allObjects))


# this part uses the distributive law to put the KB in conjunctive normal form


# this checks an object for children inside it that are of the same type and incorporantes the contents of its child in itself

def distribute(object):
    if isinstance(object,Or):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            if isinstance(obj,Or):
                for objj in obj.allObjects:
                    object.add(objj)
                object.allObjects.pop(i)
                i = -1
            i += 1
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            obj = distribute(obj)
            object.allObjects[i] = obj
            i += 1
    elif isinstance(object,And):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            if isinstance(obj,And):
                for objj in obj.allObjects:
                    object.add(objj)
                object.allObjects.pop(i)
                i = -1
            i += 1
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            obj = distribute(obj)
            object.allObjects[i] = obj
            i += 1
    elif isinstance(object,KnowledgeBase):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            if isinstance(obj,And):
                for objj in obj.allObjects:
                    object.add(objj)
                object.allObjects.pop(i)
                i = -1
            i += 1
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            obj = distribute(obj)
            object.allObjects[i] = obj
            i += 1
    return object

KB = distribute(KB)
print(KB.formula())

# this actually performs the redistribution
def redistribute(object):
    # print("entering:",a,"\t",type(object),"\t",object.formula())
    object = distribute(object)
    if isinstance(object,Or):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            if isinstance(obj,And):
                j = 0
                while j < len(object.allObjects):
                    objj = object.allObjects[j]
                    if i != j:
                        if isinstance(objj,And):
                            newPck = []
                            for objk in objj.allObjects:
                                newPck.append(Or(objk,obj))
                        else:
                            newPck = []
                            for objk in obj.allObjects:
                                newPck.append(Or(objk,objj))
                        break
                    j += 1
                obj = And(newPck)
                obj = distribute(obj)
                # print("new pack:",obj.formula())
                object.allObjects[i] = obj
                object.allObjects.pop(j)
                i = -1
            i += 1
            # print("this point:",object.formula())
            object = distribute(object)
            if len(object.allObjects) == 1:
                # print("entered as",type(object))
                object = object.allObjects[0]
                # print("exited as",type(object))
                break
        object = distribute(object)
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            obj = redistribute(obj)
            object.allObjects[i] = obj
            i += 1
    elif isinstance(object,KnowledgeBase):
        i = 0
        while i < len(object.allObjects):
            object.allObjects[i] = redistribute(object.allObjects[i])
            i += 1
    object = distribute(object)
    # print("returning:",a,"\t",type(object),"\t",object.formula())
    return object

E = Symbol("e")
kk = And(A,B)
KB = KnowledgeBase(kk)
print(KB.formula())
KB = redistribute(KB)
print(KB.formula())
print(type(KB))


def clean(KB):
    added = False
    i = 0
    # print("weh")
    while i < len(KB.allObjects):
        object = KB.allObjects[i]
        j = 0
        # print("weh")
        while j < len(KB.allObjects):
            objj = KB.allObjects[j]
            # print("mmh")
            if i != j:
                # print(distribute(Not(object)).formula())
                # print(objj.formula())
                if objj.formula() == distribute(Not(object)).formula():
                    KB.add(True)
                    # print("found")
                    added = True
                    break
            j += 1
        if added:
            break
        i += 1
    if not added:
        i = 0
        while i < len(KB.allObjects):
            object = KB.allObjects[i]
            if isinstance(object,Not):
                j = 0
                while j < len(KB.allObjects):
                    objj = KB.allObjects[j]
                    if i != j and isinstance(objj,Or):
                        k = 0
                        while k < len(objj.allObjects):
                            objk = objj.allObjects[k]
                            if objk.formula() == distribute(Not(object)).formula():
                                objj.allObjects.pop(k)
                                added = True
                                if len(objj.allObjects) == 1:
                                    KB.allObjects[j] = objj.allObjects[0]
                                break
                    j += 1
            elif isinstance(object,Symbol):
                j = 0
                while j < len(KB.allObjects):
                    objj = KB.allObjects[j]
                    if i != j and isinstance(objj,Or):
                        k = 0
                        while k < len(objj.allObjects):
                            objk = objj.allObjects[k]
                            if objk.formula() == distribute(Not(object)).formula():
                                objj.allObjects.pop(k)
                                added = True
                                if len(objj.allObjects) == 1:
                                    KB.allObjects[j] = objj.allObjects[0]
                                break
                            k += 1
                    j += 1
            i += 1
            if added:
                KB = correct(KB)
    return KB


def simplify(object):
    if isinstance(object,Symbol):
        return object
    elif isinstance(object,Not):
        return object
    elif isinstance(object,And):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            j = i
            while j < len(object.allObjects):
                obj2 = object.allObjects[j]
                rm = []
                if i != j:
                    if obj2.formula() == obj.formula():
                        rm.append(j)
                    elif obj2.formula() == distribute(Not(obj)).formula():
                        return False
                j += 1
            while len(rm) != 0:
                object.allObjects.pop(rm.pop())
                i = -1
            i += 1
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            object.allObjects[i] = simplify(obj)
            i += 1
        return object
    elif isinstance(object,Or):
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            j = i
            while j < len(object.allObjects):
                obj2 = object.allObjects[j]
                rm = []
                if i != j:
                    if obj2.formula() == obj.formula():
                        rm.append(j)
                    elif obj2.formula() == distribute(Not(obj)).formula():
                        return True
                j += 1
            while len(rm) != 0:
                object.allObjects.pop(rm.pop())
                i = -1
            if len(object.allObjects) == 0:
                return True
            i += 1
        i = 0
        while i < len(object.allObjects):
            obj = object.allObjects[i]
            object.allObjects[i] = simplify(obj)
            i += 1
        return object


def checkEntailment(KB,query):
    KB.add(Not(query))
    KB = removeBiconditional(KB)
    KB = removeImplications(KB)
    KB = notsInside(KB)
    KB = redistribute(KB)
    print(KB.formula())
    print(len(KB.allObjects))
    KB = clean(KB)
    i = 0
    while i < len(KB.allObjects):
        if isinstance(KB.allObjects[i],bool):
            return True
        i += 1
    return False


kk = Or(Not(B),And(A,B))
KB = KnowledgeBase(kk)
print(checkEntailment(KB,C))