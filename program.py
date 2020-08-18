import bootstrap
from entities.cmscontent import cmscontent

for c in cmscontent.objects.all():
    print(c)

print("where condition")
for c in cmscontent.objects.filter(UrlRef=""):
    print (c)

while True:
    
    print("Type 'quit' to exit program")
    cmd = input()

    if(cmd=='quit'):
        exit(0)  