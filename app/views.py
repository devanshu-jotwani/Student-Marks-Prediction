from django.shortcuts import render
# Create your views here.
 
from .models import Student


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier

def model1(x_train,y_train,x_test,y_test):
    #defining the model layers
    model=Sequential([
        Dense(64,activation='relu',input_dim=7),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(1,activation='sigmoid'),
        ])
        
    #compiling the model
    model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
    hist = model.fit(x_train, y_train,epochs=100, verbose=0,validation_data=(x_test,y_test))
    print(hist)
    print(model.evaluate(x_test, y_test))


def model2(x_train,y_train,x_test,y_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print("------------------")
    print(y_pred)
    print("------------------")

def splitting_inputs():
    #reading the cleaned data
    df=pd.read_csv("studentdata1.csv")

    #dividing the column into x and y
    x=df[['10th','12th','sem3','sem2','sem1','sem5','sem4']]
    y=df[['sem6']]

    '''
    Preprocessing data. Scaling input data so
    that all the input data lie between 0 to 1
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(x)

    #print(X_scale[0:10])
    #print(y)

    #splitting data into train ,test and validation data
    x_train,x_test,y_train,y_test=train_test_split(X_scale,y, train_size=100)
    #print(x_train.shape,x_test.shape)

    return (x_train,x_test,y_train,y_test)

def home(request):
    x_train,x_test,y_train,y_test=splitting_inputs()
    return render(request,"home.html",{})

def analysis(request):
    context={}
    return render(request,"analysis.html",context)

def predict(request):
    context={}
    if request.method=="POST":
        tenthmark=request.POST["tenth"]
        twelth=request.POST['twelth']
        sem1=request.POST['sem1']
        sem2=request.POST['sem2']
        sem3=request.POST['sem3']
        sem4=request.POST['sem4']
        sem5=request.POST['sem5']
        sem6=request.POST['sem6']
        lassiscore=request.POST['lassi']

        return render(request,"predict.html",context)
    else:
        return render(request,"predict.html",{})
