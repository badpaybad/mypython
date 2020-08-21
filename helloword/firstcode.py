import sys
a="hello"
def myfunc():
    print("xxxx")
    a="du"
    print (a)

myfunc()

print(a)

x="xxxxx\""
print(x, x[3:9])

print (str.format("abc {}",a))

for x in range(10):
    if x%2 == 0: 
        print(x)
    else:
        pass

# while True:
#     print(a)

powOf = lambda a : a*a

print(powOf(4))

def lbdInside(n):
    return lambda a:a+n

print (lbdInside(3)(5))

class Xxx:
    Name=""
    def __init__(self):
        self.Name="hello"

x = Xxx()
print (x.Name)

import helloworld

print(helloworld)

print(dir(helloworld))

import datetime 

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

import json
print(json.dumps({"A":"a"}))

print(json.dumps(("a","b")))

print(json.dumps([1,"a"]))

print(json.dumps({"a","b"}))

import threading, queue