number = 3

states = []

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

print(generate(1,number))