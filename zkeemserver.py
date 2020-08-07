import socket
import sys

class HttpRequest:
    def __init__(self)
        self.method=""
        self.header=""
        self.body=""
    def parse(self, strReceived):
        #parse your received string
        method, url, header,body= strReceived
        return method, url, header, body
class HttpResponse:
    def CdataPost(self, request: HttpRequest):
        header,body=""
        return header, body    
    def CdataGet(self, request: HttpRequest):
        header,body=""
        return header, body    
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.2.44', 8081)
sock.bind(server_address)
sock.listen(100)
while True:
    conn, client_address = sock.accept() # wait next request comming
    #process current request
    print("Client address: "+ client_address)
    requestData=[]
    while True:
        temp = conn.recv(1024)
        if not temp: 
            break
        requestData.append(temp)
    #process requestData
    requestInString = ''.join(map(chr,requestData))
    objRequest = new HttpRequest().parse(requestInString)
    objResponse=null
    if objRequest.url.index("iclock/cdata")==0:
        if objRequest.method=="get" :
            objResponse= new HttpResponse().CdataGet(objRequest)
        else
            objResponse= new HttpResponse().CdataPost(objRequest)    
    #process response
    if objResponse!=null:
        conn.send(objResponse.header)
        conn.send(objResponse.body)
    
    conn.close()

sock.close()
