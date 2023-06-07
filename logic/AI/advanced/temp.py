def removeNots(object):
    if isinstance(object,Symbol):
        return Not(object)
    elif isinstance(object,Not):
        object = removeNots(object.object)
        if isinstance(object,Not):
            object = object.object
    elif isinstance(object,And):
        newObj = []
        for obj in object.allObjects:
            newObj.append(removeNots(Not(obj)))
        object = Or(newObj)
    elif isinstance(object,Or):
        newObj = []
        for obj in object.allObjects:
            newObj.append(removeNots(Not(obj)))
        object = And(newObj)
    return object