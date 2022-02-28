# import the standard Django Model

from django.db import models
   

class Student(models.Model):
  
    # fields of the model
    title = models.CharField(max_length = 200)
    tenth = models.FloatField(default=0)
    twelth = models.FloatField(default=0)
    sem1 = models.FloatField(default=0)
    sem2 = models.FloatField(default=0)
    sem3 = models.FloatField(default=0)
    sem4 = models.FloatField(default=0)
    sem5 = models.FloatField(default=0)
    sem6 = models.FloatField(default=0 )

  
    
    def __str__(self):
        return self.title