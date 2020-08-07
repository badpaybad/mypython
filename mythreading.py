import threading;
import multiprocessing;

class Test:
    def __init__(self,name:str="") :
        self.name=name

    def changeName(self,name:str):
        self.name= name
        return self

test =Test("Hi").changeName("Du")

print (test.name)


map={}
map["x"]= lambda x : print(x)
map["y"]= lambda x: Test(x).name

print (map["y"]("Abc"))