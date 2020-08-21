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
        self.urlQueryString = ""
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
            "urlQueryString": self.urlQueryString,
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
        self.urlQueryString = urlParams[1] if len(urlParams) > 1 else ""
        self.method = firstLine[0].upper()
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

    def RegisterHanle(self, url:str, func):
        if url.startswith("/")==False :
            url="/"+url
        url=url.lower()
        self.__routing[url]=func

    def Handle(self, request: HttpRequest):
        if (request.url in self.__routing):
            return self.__routing[request.url](request)
        return HttpResponse().NotFound404(request)

    def Index(self, request: HttpRequest):
        return HttpResponse().JsonContent({"version": "0.0.1"})
    
    def Process_cdata(self,request: HttpRequest):
        res= HttpResponse().JsonContent({"cdata":"cdata", "queryString": request.urlQueryString})
        res.header=res.header+f"CustomHeader: python-response\r\n"
        #process data to return attendance device
        return res      

class HttpServerSocket:

    def __init__(self, routingHandle:RoutingHandle):
        self._hostOrDomain="localhost"
        self._port=8088
        self._routingHanle=routingHandle
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)      

        self._isStopListenter=False
        self._isEnded=False
        self._mainThread: threading.Thread
        
        self._mainThread=threading.Thread(target=self.__processSocketAccepted, args=(self._sock,) , daemon=True)

    def __socketHandleRequest(self,conn: socket.socket, clientAddress):
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

            objResponse = self._routingHanle.Handle(objRequest)

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

    def __processSocketAccepted (self,currentSock:socket.socket):   
        while not self._isStopListenter:
            try:
                client_conn, client_address = currentSock.accept()  # wait next request comming
                print(f"Connected Client address: {client_address}")
                # process current request
                currentConnectedThread=threading.Thread(target=self.__socketHandleRequest, args=(client_conn,client_address,))
                currentConnectedThread.daemon=True
                currentConnectedThread.start()

            except Exception as ell:
                print(json.dumps({"code": "0", "message": f"{ell}"}))
        
        self._isEnded=True
            
    """
    # need help : got error OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
    # threading no error but multiprocessing got above error
    """
    #multiprocessing this no work : OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
    #sw = multiprocessing.Process(target=processSocketAccepted, args=(_sock,), daemon=True)
    #sw.start()
    #threading: this worked well    
   
    """ 
    _socketWorkers=[]

    for i in range(0):
        sw = multiprocessing.Process(target=processSocketAccepted, args=(_sock,))
        _socketWorkers.append(sw)

    for sw in _socketWorkers:
        sw.daemon=True
        sw.start()
    """
    def Start(self, hostOrDomain:str="localhost", port:int=8088):
        self._hostOrDomain= hostOrDomain
        self._port=port
        
        self._server_address = (self._hostOrDomain, self._port)
        self._sock.bind(self._server_address)
        self._sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
        self._sock.listen(1000) #is pool size (max concurrent connection???)
        self._sock.settimeout(120)

        self._mainThread.start()

        print(f"Started socket: {self._server_address}:{self._port}")

    def Stop(self):
        self._isStopListenter=True
        
        try:
            print("Call to stop socket listener")
            _callToClose = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _callToClose.connect((self._hostOrDomain,self._port))    
        except:
            pass
        #_mainThread.join()
        
        while not self._isEnded:
            time.sleep(1)

        self._sock.close()   
        print("Stoped socket")

#dynamic register url routing
#__globalRoutingHandle.RegisterHanle("Test", lambda r: FunctionForRoutingTest(r) )

#def FunctionForRoutingTest(request:HttpRequest):
#    return HttpResponse().JsonContent({"Test":"Test"})

__globalRoutingHandle = RoutingHandle()
__globalHttpServer =HttpServerSocket(__globalRoutingHandle)