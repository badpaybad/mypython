import socket
import sys
import threading
import queue
import time
import datetime
import multiprocessing
import os
import json
from typing import Any, cast

class Contants:
    _splitCr = '\r'
    _splitNewLine = '\n'
    _splitQuery = '?'
    _splitAnd = '&'
    _splitEqual = '='
    _splitSpeace = ' '
    _splitSlash = '/'
    _splitColon = ':'

class HttpRequest:
    
    def __init__(self, strReceived=""):
        self.httpVersion = ""
        self.method = ""
        self.header = ""
        self.body = ""
        self.url = ""
        self.urlFull = ""
        self.urlParam = ""
        self.raw = strReceived
        self.__parse(strReceived)

    def ToJson(self):
        return json.dumps({
            "httpVersion": self.httpVersion,
            "method": self.method,
            "header": self.header,
            "body": self.body,
            "url": self.url,
            "urlFull": self.urlFull,
            "urlParam": self.urlParam,
            "raw": self.raw
        })

    def __parse(self, strReceived):
        if strReceived=="": 
            return self
        # parse your received string
        lines = strReceived.split(Contants._splitNewLine)
        firstLine = lines[0].split(Contants._splitSpeace)
        self.httpVersion = firstLine[2].strip(Contants._splitCr).strip(
            Contants._splitNewLine).strip(Contants._splitSpeace)
        self.urlFull = firstLine[1]
        urlParams = self.urlFull.split(Contants._splitQuery)
        self.url = urlParams[0].lower()
        self.urlParam = (urlParams[1] if len(urlParams) > 1 else "").lower()
        self.method = firstLine[0].lower()
        beginBody = 0
        linesCount = len(lines)
        for i in range(1, linesCount):
            l = lines[i].strip(Contants._splitCr).strip(Contants._splitNewLine)
            self.header = self.header + l+"\n"
            if(not l):
                beginBody = i
                break

        for i in range(beginBody, linesCount):
            self.body = self.body + lines[i]+"\n"

        self.body = self.body.strip(Contants._splitCr).strip(Contants._splitNewLine)
        return self


class HttpResponse:  

    def __init__(self, request: HttpRequest=Any):
        self.header = ""
        self.body = ""
        self.httpStatus = "200"
        self._request = cast(HttpRequest, request) if isinstance(request, HttpRequest) else  HttpRequest()

    def JsonContent(self, obj):
        self.body = json.dumps(obj)
        self.httpStatus = "200"
        return self

    def NotFound404(self, request: HttpRequest):
        self.httpStatus = "404"
        self.body = "404 Not found"
        return self

    def ToJson(self):
        return json.dumps({
            "header": self.header,
            "body": self.body,
            "httpStatus": self.httpStatus
        })

class RoutingHandle:
    def __init__(self):
        self.__routing = {}
        self.__routing[""] = self.Index
        self.__routing["/"] = self.Index
        self.__routing["/iclock/cdata"]= self.Process_cdata

    def Handle(self, request: HttpRequest):
        if (request.url in self.__routing):
            return self.__routing[request.url](request)
        return HttpResponse().NotFound404(request)

    def Index(self, request: HttpRequest):
        return HttpResponse().JsonContent({"version": "0.0.1"})
    
    def Process_cdata(self,request: HttpRequest):
        res= HttpResponse().JsonContent({"cdata":"cdata"})
        res.header=res.header+f"CustomHeader: python-response\r\n"
        #process data to return attendance device
        return res      


def socketHandleRequest(conn: socket.socket, clientAddress):
    try:
        requestData = []
        
        #not sure when use "while True" and buffer 1024 always got disconnected from client
        #[WinError 10053] An established connection was aborted by the software in your host machine
        #while True:
        #    temp = conn.recv(1024, 0) 
        #    if not temp:
        #        break
        #    requestData.append(temp)
        
        requestData.append(conn.recv(4096000000, 0))#trick get big data at one times
        
        # process requestData
        requestInString = ''.join(map(lambda x: str(x, "utf-8"), requestData))

        objRequest = HttpRequest(requestInString)
        
        print('Request:\r\n')
        print(requestInString)

        objResponse = RoutingHandle().Handle(objRequest)

        # process response
        if not objResponse:
            objResponse = HttpResponse().NotFound404(objRequest)

        bodyInBytes = bytes(objResponse.body, "utf-8")        
        tempResponseHeader = objResponse.header       
        objResponse.header = ""
        objResponse.header = objResponse.header + objRequest.httpVersion+" "+objResponse.httpStatus+"\r\n"
        objResponse.header = objResponse.header+"Server: HttpPyHost-v1\r\n"
        objResponse.header = objResponse.header+"Content-Type: application/json\r\n"        
        objResponse.header = objResponse.header + f"Content-Length: {len(bodyInBytes)}\r\n"
        objResponse.header = objResponse.header+tempResponseHeader     
        objResponse.header = objResponse.header+"Connection: close\r\n\r\n"        

        try:
            print('Response:\r\n')
            print(objResponse.header)
            print(objResponse.body)
            conn.sendall(bytes(objResponse.header, "utf-8"), 0)
            conn.sendall(bodyInBytes, 0)
        except Exception as e:
            print(json.dumps({"code": "0", "message": f"{e}"}))
    except Exception as ex:
        print(f"{ex}")
    finally:         
        conn.close()

_hostOrDomain="localhost"
_port=8081

_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_server_address = (_hostOrDomain, _port)
_sock.bind(_server_address)
_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
_sock.listen(1000) #is pool size (max concurrent connection???)
_sock.settimeout(120)

print(f"Listening {_server_address}")

_isStopListenter=False

def socketWorker (currentSock:socket.socket):   
    while not _isStopListenter:
        try:
            client_conn, client_address = currentSock.accept()  # wait next request comming
            print(f"Connected Client address: {client_address}")
            # process current request
            currentConnectThread=threading.Thread(target=socketHandleRequest, args=(client_conn,client_address,))
            currentConnectThread.daemon=True
            currentConnectThread.start()

        except Exception as ell:
            print(json.dumps({"code": "0", "message": f"{ell}"}))
        
"""
# need help : got error OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
# threading no error but multiprocessing got above error
_socketWorkers=[]

for i in range(10):
    sw = multiprocessing.Process(target=socketWorker, args=(_sock,))
    _socketWorkers.append(sw)

for sw in _socketWorkers:
    sw.daemon=True
    sw.start()
"""

_mainThread=threading.Thread(target=socketWorker, args=(_sock,) , daemon=True)
_mainThread.start()

while True:
    cmd = input()
    if cmd =="quit":               
        break
    else :
        time.sleep(1)

_isStopListenter=True

try:
    print("Call to stop socket listener")
    _callToClose = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _callToClose.connect((_hostOrDomain,_port))    
except:
    pass

_mainThread.join()

_sock.close()   

print("Stoped socket")
