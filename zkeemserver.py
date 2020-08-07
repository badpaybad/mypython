import socket
import sys


class HttpRequest:
    def __init__(self, strReceived):
        self.method = ""
        self.header = ""
        self.body = ""
        self.url = ""
        self.raw = strReceived
        self.parse(strReceived)

    def parse(self, strReceived):
        # parse your received string
        self.method = ""
        self.header = ""
        self.body = ""
        self.url = ""
        return self


class HttpResponse:
    def __init__(self):
        self.header = ""
        self.body = ""

    def CdataPost(self, request: HttpRequest):
        self.header = ""
        self.body = "OK"
        return self

    def CdataGet(self, request: HttpRequest):
        self.header = ""
        self.body = "OK"
        return self


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.2.44', 8081)
sock.bind(server_address)
sock.listen(100)
while True:
    conn, client_address = sock.accept()  # wait next request comming
    # process current request
    print("Client address: " + client_address)
    requestData = []
    while True:
        temp = conn.recv(1024)
        if not temp:
            break
        requestData.append(temp)
    # process requestData
    requestInString = ''.join(map(chr, requestData))
    objRequest = HttpRequest(requestInString)
    objResponse = None
    if objRequest.url.index("iclock/cdata") == 0:
        if objRequest.method == "get":
            objResponse = HttpResponse().CdataGet(objRequest)
        else:
            objResponse = HttpResponse().CdataPost(objRequest)
    # process response
    if objResponse != None:
        conn.send(bytes(objResponse.header, "utf-8"))
        conn.send(bytes(objResponse.body, "utf-8"))

    conn.close()

sock.close()
