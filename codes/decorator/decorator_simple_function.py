import math

def check(anyf):
    def anyname(c, d):
        if d == 0:
            print('you are xxxxed')
            return math.nan
        return anyf(c, d)
    return anyname

# method 1
#def div(a,b):
#    return a / b
#div = check(div)

# method 2
@check
def div(a,b):
    return a / b

print(div(10,0))
print(div(10,2))

