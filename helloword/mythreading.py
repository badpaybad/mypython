import threading
import queue
import time
import datetime
import multiprocessing


class Test:
    def __init__(self, name: str = ""):
        self.name = name

    def changeName(self, name: str):
        self.name = name
        return self


test = Test("Hi").changeName("Du")

print(test.name)


myqueue = queue.Queue()


def ConcurrentWorker():
    semaphore = threading.Semaphore(2)
    semaphoreCounter = 0
    while True:
        itm = myqueue.get()
        if not itm:
            time.sleep(1)
            continue

        if semaphore.acquire(False):
            try:
                semaphoreCounter = semaphoreCounter+1
                print(f"semaphore {semaphoreCounter} {itm}")
            finally:
                semaphore.release()


def EnqueueWorker():
    while True:
        myqueue.put(datetime.datetime.now())
        time.sleep(0.4)


mythreadDequeue = threading.Thread(target=ConcurrentWorker, daemon=True)
mythreadDequeue.start()

mythreadEnqueue = threading.Thread(target=EnqueueWorker, daemon=True)
mythreadEnqueue.start()


map = {}
map["x"] = lambda x: print(x)
map["y"] = lambda x: Test(x).name

print(map["y"]("Abc"))

none = None

print(none == None)
print(none == 'None')

print("Type 'quit' to exit")
cmd = input()
