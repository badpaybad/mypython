import bootstrap
from entities.cmscontent import cmscontent
import libs.HttpPyHost
from libs.HttpPyHost import HttpResponse

if( __name__ =="__main__"):
    # register your routing handle
    libs.HttpPyHost.__globalRoutingHandle.RegisterHanle("/test", lambda r: libs.HttpPyHost.HttpResponse().JsonContent({"id":"test"}))
    
    #start your code here

    libs.HttpPyHost.__globalHttpServer.Start()
    while True:
        cmd=input()
        if(cmd=='quit'):
            libs.HttpPyHost.__globalHttpServer.Stop()
            break   
        
        print("Type 'quit' to exit")
