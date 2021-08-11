from multiprocessing import Process
import os
import time
from datetime import datetime

class Test:
    def __init__(self) -> None:
        time.sleep(3)
        self.Name = "now: {}".format(datetime.now())
    def Change(self, name):
        print("-----------------{}".format(name))
        print("changed: {}".format(datetime.now()))
        print("name: {}".format(self.Name))
    def Show(self):
        print("------{}".format( self.Name))
test = Test()

if __name__ == '__main__':

    p = Process(target=test.Change, args=("process 1",))
    p.start()
    p.join()
    p1 = Process(target=test.Change, args=("process 2",))
    p1.start()  
    p1.join()

    test.Change("same thread")
    test.Show()
