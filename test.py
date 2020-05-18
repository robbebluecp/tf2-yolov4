



def go():
    for i in range(10):
        yield i, i+1


for x, y in go():
    print(x, y)
