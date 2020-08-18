import django
import os
from django.conf import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(BASE_DIR)

settings.configure(
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'moneynote',
            'HOST': 'localhost',
            'PORT': '3306',
            'USER': 'root',
            'PASSWORD': '',
        }
    },
    INSTALLED_APPS=[
        'entities',
        
    ]
)

django.setup()