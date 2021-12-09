def getnearest(el, coll, func):
    res = [x for x in coll if func(el, x)]
    for x in res:
        coll.remove(x)
    return res 
    
def getclusterrecurs(el, coll, func):
    cl = []
    nearest = getnearest(el, coll, func)
    
    if len(nearest) == 0:
        return cl
    
    cl.extend(nearest)
    for n in nearest:
        m = getclusterrecurs(n, coll, func)
        cl.extend(m)
    
    return cl
    
def clusterize(x, func):
    clusters = []    
    
    objectslist = list(x)
    
    while len(objectslist) > 0:
    
        first = objectslist.pop(0)
        cl = getclusterrecurs(first, objectslist, func)
        cl.append(first)
        clusters.append(cl)
    
    return clusters
    
