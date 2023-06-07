from time import time


a = [1,2,3,4,5,6,7,8,9,10]
stop = time()
i = 0

def funct(num,a):
    for a1 in a:
        if a1 == num:
            return True
    return False

def trial(num,a):
    try:
        a.remove(num)
    except:
        return False
    a.append(num)
    return True

count = 0
while i < 100000:
    start = time()
    val = funct(5,a) #fast
    val = 5 in a #fastest
    val = trial(5,a) #faster
    val =  not all(a1 != 5 for a1 in a) #slowest
    if val:
        stop = time()
    count+=(stop - start)
    i += 1
print(count/100000)