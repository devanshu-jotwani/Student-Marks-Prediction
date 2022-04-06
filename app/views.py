from django.shortcuts import render
# Create your views here.
 
from .models import Student


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse

#DATA PROCESSING
#reading the cleaned data
df=pd.read_csv("../studentdata1.csv")

#dividing the column into x and y
x=df[['10th','12th','sem3','sem2','sem1','sem5','sem4']]
y=df[['sem6']]

'''
Preprocessing data. Scaling input data so
that all the input data lie between 0 to 1
'''
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(x)


#splitting data into train ,test and validation data
x_train,x_test,y_train,y_test=train_test_split(X_scale,y, train_size=100)
#print(x_train.shape,x_test.shape)



def model1(x_train,y_train,x_test,y_test):
    #defining the model layers
    #model=NN
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
model1(x_train,y_train,x_test,y_test)

def model2(x_train,y_train,x_test,y_test):
    #model= DTR 
    DTRModel = DecisionTreeRegressor()
    DTRModel.fit(x_train, y_train)

    y_pred = DTRModel.predict(x_test)
    #result=pd.DataFrame({"Actual:":y_test,"Predicted":y_pred})
    #print(result)

    print("Mean Squared Error ",mse(y_test,y_pred))
    print("Mean Absolute Error ",metrics.median_absolute_error(y_test,y_pred))
model2(x_train,y_train,x_test,y_test)


def home(request):

    return render(request,"home.html",{})


def analysis(request):
    context={}
    return render(request,"analysis.html",context)

def predict(request):
    context={}
    error=False
    if request.method=="POST":
        tenthmark=request.POST["tenth"]
        twelth=request.POST['twelth']
        sem1=request.POST['sem1']
        sem2=request.POST['sem2']
        sem3=request.POST['sem3']
        sem4=request.POST['sem4']
        sem5=request.POST['sem5']
        sem6=request.POST['sem6']
        if twelth>100 or twelth<0 and tenthmark>100 or tenthmark<0:
            error=True
        if sem1>10 or sem1<0 and sem2>10 or sem2<0 and sem3>10 or sem3<0 and sem4>10 or sem4<0 and sem5>10 or sem5<0 and sem6>10 or sem6<0:
            error=True
        if error:
            return render(request,"predict.html",{error:error})
        else:
            return render(request,"predict.html",context)
    else:
        return render(request,"predict.html",{})
