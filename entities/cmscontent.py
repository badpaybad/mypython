import django
from django.db import models


class cmscontent(models.Model):

    Id = models.CharField(db_column="Id", max_length=36, primary_key=True)

    Title = models.TextField(db_column="Title")

    UrlRef =  models.TextField()
    
    class Meta:
        managed = False
        db_table = 'cmscontent'

    def __str__(self):
        return f"{self.Id}:{self.Title}"
