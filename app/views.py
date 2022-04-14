
from fileinput import filename
from django.shortcuts import render,redirect

from sklearn.ensemble import RandomForestRegressor
# Create your views here.
 
from .models import Student

import joblib
import pandas as pd
from sklearn import naive_bayes, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

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
#print("Before Scaling \n",x)
#print("After Scaling \n",X_scale)

#splitting data into train ,test and validation data
x_train,x_test,y_train,y_test=train_test_split(X_scale,y, train_size=100)
#print(x_train.shape,x_test.shape)


#Neural Network Model
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
    y_pred=model.predict(x_test)
    #print("NN Prediction- ",y_pred)
    #print(model.evaluate(x_test, y_test))
    filename="NN.sav"
    joblib.dump(model,filename)
    print("Neural Network Prediction")
    print(y_pred)
    return y_pred
#NNpred=model1(x_train,y_train,x_test,y_test)

#Decission Tree Regressor Model
def model2(x_train,y_train,x_test,y_test):
    #model= DTR 
    DTRModel = DecisionTreeRegressor()
    DTRModel.fit(x_train, y_train)
    # save the model to disk
    filename = 'DTR.sav'
    joblib.dump(DTRModel, filename)

    y_pred = DTRModel.predict(x_test)
    print("Predicted DTR Model")
    return y_pred    
DTRpred=model2(x_train,y_train,x_test,y_test)

#Linear Regression Model
def model3(x_train,y_train,x_test):
    #model=LR
    LRModel=LinearRegression()
    LRModel.fit(x_train,y_train)
     
    y_pred=LRModel.predict(x_test)
    y_pred=[round(item,2) for sublist in y_pred for item in sublist]
    filename='LR.sav'
    joblib.dump(LRModel,filename)
    
    print("Predicted Linear regression")
    
    return y_pred
LRpred=model3(x_train,y_train,x_test)


#Random Forest Regressor
def model4(x_train,y_train,x_test):
    #random forest
    RFRModel=RandomForestRegressor(n_estimators=10,random_state=0)
    RFRModel.fit(x_train,y_train)
    y_pred=RFRModel.predict(x_test)
    print(y_pred)
    filename='RFR.sav'
    joblib.dump(RFRModel,filename)
    print("Predicted Random Forest Regressor")
    return y_pred
RFRpred=model4(x_train,y_train,x_test)

#Naive Bayes Model
def model5(x_train,y_train,x_test):
    #Naive Bayes Classifier
    result=0
    NBModel=0
    NBModel.fit(x_train,y_train)
    y_pred=NBModel.predict(x_test)

    filename='NB.sav'
    joblib.dump(NBModel,filename)
    return y_pred

#NBpred=model5(x_train,y_train,x_test)

#Decision tree classifier
def model6(x_train,y_train,x_test):
    #Decision Tree Classifier
    
    DTCModel=DecisionTreeClassifier()
    DTCModel.fit(x_train,y_train)
    y_pred=DTCModel.predict(x_test)
    filename='DTC.sav'
    joblib.dump(DTCModel,filename)
    return y_pred
#DTCpred=model6(x_train,y_train,x_test)


#Home Page view function
def home(request):

    return render(request,"index.html",{})


#Analyses Page view function
def analysis(request):
    from sklearn import metrics
    context={
        'Predictions':
        [
        {
            "Model":"Decision Tree Regressor",
            "Accuracy":round(metrics.r2_score(y_test, DTRpred)*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test, DTRpred),2),
            "RootMeanSquaredError":round(np.sqrt(metrics.mean_squared_error(y_test, DTRpred)),2)
        },
        {
            "Model":'Linear Regression',
            "Accuracy":round(metrics.r2_score(y_test, LRpred)*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test, LRpred),2),
            "RootMeanSquaredError":round(np.sqrt(metrics.mean_squared_error(y_test, LRpred)),2)
        },
        {
            "Model":"Random Forest Regressor",
            "Accuracy":round(metrics.r2_score(y_test, RFRpred)*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test, RFRpred),2),
            "RootMeanSquaredError":round(np.sqrt(metrics.mean_squared_error(y_test, RFRpred)),2)
        }
        
        ]
        ,"Classification":
        [    
                {
                    "Model":"Decision Tree Classifier",
                    "Accuracy":"",
                    "ConfusionMatrix":"",
                    "ClassificationReport":""
                },
                {
                    "Model":"Naive Bayes ",
                    "Accuracy":"",
                    "ConfusionMatrix":"",
                    "ClassificationReport":""
                }
        ]
            
    }
    
    return render(request,"analysis.html",context)

def predict(request):
    
    context={}
    error=False
    if request.method=="POST":
        tenthmark=float(request.POST["tenth"])
        twelth=float(request.POST['twelth'])
        sem1=float(request.POST['sem1'])
        sem2=float(request.POST['sem2'])
        sem3=float(request.POST['sem3'])
        sem4=float(request.POST['sem4'])
        sem5=float(request.POST['sem5'])
        sem6=float(request.POST['sem6'])
        if twelth>100 or twelth<0 and tenthmark>100 or tenthmark<0:
            error=True
        if sem1>10 or sem1<0 and sem2>10 or sem2<0 and sem3>10 or sem3<0 and sem4>10 or sem4<0 and sem5>10 or sem5<0 and sem6>10 or sem6<0:
            error=True
        if error:
            return render(request,"prediction.html",{error:error})
        else:
            #Transforming input data
            unseenData=[[tenthmark,twelth,sem1,sem2,sem3,sem4,sem5]]
            eval_X=min_max_scaler.fit_transform(unseenData)
            prediction=list(analyze_unseen_data(eval_X,sem6))
            print(prediction)
            return render(request,"prediction.html",{'prediction':float(prediction[1]),'actual':sem6})
    else:
        return render(request,"prediction.html",{})


def analyze_unseen_data(eval_x,output):
    result=0
    #loading prediction models
    DTRModel=joblib.load('DTR.sav')
    LRModel=joblib.load("LR.sav")
    
    result=DTRModel.predict(eval_x)
    result2=LRModel.predict(eval_x)
    #loading classifier models
    #NBModel=joblib.load('NB.sav')
    #DTCModel=joblib.load('DTC.sav')
    return (result,result2)


#table of actual and predicted value
def table_view(request):
    import json
    
    test=y_test['sem6'].tolist()
    print(type(test))
    print(type(LRpred))
    df = pd.DataFrame({'ActualValue':test,"LR":list(LRpred),"RFR":list(RFRpred),"DTR":list(DTRpred)})
    
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'table': data}
    return render(request,"table.html",context)

#graph page
def graph_view(request):

    n=len(y_test)
    x_axis=list(range(1,n+1))
    import json
    #xaxis data
    x_axis=json.dumps(x_axis)
    #test data
    
    test=json.dumps(y_test['sem6'].tolist())
    
    context={'x_axis':x_axis,'test_output':test,'lr_output':list(LRpred),"rfr_output":list(RFRpred)
    #,"nn_output":list(NNpred)
    ,"dtr_output":list(DTRpred)}
    return render(request,"graph.html",context)